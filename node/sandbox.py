"""Restricted local command execution for DAIN NODE-4.

The node never invokes a shell.  Linux nodes additionally execute through
bubblewrap with a private network namespace and only the scratch directory
writable.  A deliberately small command allowlist keeps the non-Linux
development fallback useful without pretending it is a general-purpose jail.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path, PurePath
from threading import Lock, Thread
from typing import Any, BinaryIO, Literal

DEFAULT_SCRATCH_ROOT = Path("/var/tmp/dain")
SCRATCH_ROOT_ENV = "DAIN_SCRATCH_ROOT"
DEFAULT_TIMEOUT_S = 5.0
MAX_TIMEOUT_S = 30.0
DEFAULT_OUTPUT_CAP_BYTES = 64 * 1024
MAX_OUTPUT_CAP_BYTES = 1024 * 1024
AUDIT_LOG_NAME = "exec-audit.jsonl"

# These programs read or transform data; none opens sockets or launches a
# second program.  Linux still runs them inside bubblewrap as defence in depth.
DEFAULT_ALLOWLIST = frozenset(
    {
        "cat",
        "grep",
        "head",
        "ls",
        "rg",
        "shasum",
        "sha256sum",
        "sort",
        "tail",
        "uniq",
        "wc",
    }
)
FORBIDDEN_OPTIONS = frozenset(
    {
        "--dereference-recursive",
        "--follow",
        "--pre",
        "--pre-glob",
        "--output",
        "-L",
        "-o",
        "-R",
    }
)
FORBIDDEN_OPTION_PREFIXES = ("--output=", "--pre=", "--pre-glob=")


class SandboxRejected(ValueError):
    """The request violates a sandbox policy and was not executed."""


class SandboxExecutionError(RuntimeError):
    """The isolation runtime could not start the validated command."""


@dataclass(frozen=True)
class SandboxResult:
    invocation_hash: str
    argv: list[str]
    cwd: str
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    truncated: bool
    duration_ms: float

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, **asdict(self)}


class CommandSandbox:
    def __init__(
        self,
        scratch_root: Path | str,
        *,
        allowlist: frozenset[str] = DEFAULT_ALLOWLIST,
        require_linux_isolation: bool | None = None,
        bubblewrap: str | None | Literal[False] = None,
        clock: Any = time.monotonic,
        wall_clock: Any = time.time,
    ) -> None:
        self.root = Path(scratch_root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.allowlist = allowlist
        self.require_linux_isolation = (
            sys.platform.startswith("linux")
            if require_linux_isolation is None
            else require_linux_isolation
        )
        self.bubblewrap = (
            None
            if bubblewrap is False
            else bubblewrap
            if bubblewrap is not None
            else shutil.which("bwrap")
        )
        self.clock = clock
        self.wall_clock = wall_clock
        self.audit_path = self.root / AUDIT_LOG_NAME
        self._audit_lock = Lock()

    @classmethod
    def from_environment(cls) -> CommandSandbox:
        return cls(os.getenv(SCRATCH_ROOT_ENV, str(DEFAULT_SCRATCH_ROOT)))

    def execute(
        self,
        argv: list[str],
        *,
        cwd: str = ".",
        timeout_s: float = DEFAULT_TIMEOUT_S,
        output_cap_bytes: int = DEFAULT_OUTPUT_CAP_BYTES,
        job_id: str | None = None,
    ) -> SandboxResult:
        started = self.clock()
        invocation_hash = self._invocation_hash(argv, cwd, timeout_s, output_cap_bytes)

        try:
            executable, working_directory = self._validate(
                argv, cwd, timeout_s, output_cap_bytes
            )
            command = self._isolated_command(executable, argv[1:], working_directory)
        except SandboxRejected as exc:
            self._audit(
                invocation_hash,
                argv,
                cwd,
                status="refused",
                message=str(exc),
                duration_ms=(self.clock() - started) * 1000,
                job_id=job_id,
            )
            raise

        try:
            process = subprocess.Popen(
                command,
                cwd=working_directory,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"},
                start_new_session=True,
            )
        except OSError as exc:
            duration_ms = (self.clock() - started) * 1000
            self._audit(
                invocation_hash,
                argv,
                cwd,
                status="failed",
                message=str(exc),
                duration_ms=duration_ms,
                job_id=job_id,
            )
            raise SandboxExecutionError("sandbox runtime could not start") from exc

        assert process.stdout is not None
        assert process.stderr is not None
        output = _OutputCollector(output_cap_bytes)
        readers = [
            Thread(target=output.drain, args=(process.stdout, "stdout"), daemon=True),
            Thread(target=output.drain, args=(process.stderr, "stderr"), daemon=True),
        ]
        for reader in readers:
            reader.start()

        timed_out = False
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        for reader in readers:
            reader.join()

        stdout, stderr, truncated = output.result()
        duration_ms = (self.clock() - started) * 1000
        result = SandboxResult(
            invocation_hash=invocation_hash,
            argv=list(argv),
            cwd=str(working_directory.relative_to(self.root)) or ".",
            exit_code=None if timed_out else process.returncode,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            timed_out=timed_out,
            truncated=truncated,
            duration_ms=duration_ms,
        )
        self._audit(
            invocation_hash,
            argv,
            cwd,
            status="timed_out" if timed_out else "completed",
            message=f"exit_code={result.exit_code}",
            duration_ms=duration_ms,
            job_id=job_id,
        )
        return result

    def _validate(
        self,
        argv: list[str],
        cwd: str,
        timeout_s: float,
        output_cap_bytes: int,
    ) -> tuple[str, Path]:
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(value, str) and value for value in argv)
        ):
            raise SandboxRejected("argv must be a non-empty list of strings")

        command_name = Path(argv[0]).name
        if argv[0] != command_name or command_name not in self.allowlist:
            raise SandboxRejected(f"command {argv[0]!r} is not allowlisted")
        if any(
            value in FORBIDDEN_OPTIONS or value.startswith(FORBIDDEN_OPTION_PREFIXES)
            for value in argv[1:]
        ):
            raise SandboxRejected("command requests forbidden link or helper traversal")
        if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool):
            raise SandboxRejected("timeout_s must be a number")
        if not 0 < timeout_s <= MAX_TIMEOUT_S:
            raise SandboxRejected(f"timeout_s must be between 0 and {MAX_TIMEOUT_S}")
        if isinstance(output_cap_bytes, bool) or not isinstance(output_cap_bytes, int):
            raise SandboxRejected("output_cap_bytes must be an integer")
        if not 1 <= output_cap_bytes <= MAX_OUTPUT_CAP_BYTES:
            raise SandboxRejected(
                f"output_cap_bytes must be between 1 and {MAX_OUTPUT_CAP_BYTES}"
            )

        working_directory = self._jailed_path(cwd)
        if not working_directory.is_dir():
            raise SandboxRejected("working directory does not exist")

        for argument in argv[1:]:
            self._reject_escaping_argument(argument, working_directory)

        executable = shutil.which(command_name, path="/usr/local/bin:/usr/bin:/bin")
        if executable is None:
            raise SandboxRejected(
                f"allowlisted command {command_name!r} is not installed"
            )
        if self.require_linux_isolation and self.bubblewrap is None:
            raise SandboxRejected("bubblewrap is required for isolated execution")
        return executable, working_directory

    def _jailed_path(self, value: str) -> Path:
        if not isinstance(value, str) or not value or value.startswith("~"):
            raise SandboxRejected("working directory must be relative to scratch")
        candidate = Path(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise SandboxRejected("working directory escapes scratch")
        resolved = (self.root / candidate).resolve()
        if not resolved.is_relative_to(self.root):
            raise SandboxRejected("working directory escapes scratch")
        return resolved

    def _reject_escaping_argument(self, argument: str, cwd: Path) -> None:
        if argument.startswith("~"):
            raise SandboxRejected("home-directory paths are forbidden")

        # Check both a raw operand and an option value such as --file=/etc/x.
        candidate = argument.split("=", 1)[-1]
        path = PurePath(candidate)
        if path.is_absolute() or ".." in path.parts:
            raise SandboxRejected("arguments may not address paths outside scratch")
        resolved = (cwd / candidate).resolve()
        if (cwd / candidate).exists() and not resolved.is_relative_to(self.root):
            raise SandboxRejected("arguments may not follow links outside scratch")

    def _isolated_command(
        self, executable: str, arguments: list[str], cwd: Path
    ) -> list[str]:
        if self.bubblewrap is None:
            return [executable, *arguments]

        command = [
            self.bubblewrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-net",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--bind",
            str(self.root),
            str(self.root),
            "--chdir",
            str(cwd),
        ]
        for system_path in ("/usr", "/bin", "/lib", "/lib64"):
            if Path(system_path).exists():
                command.extend(("--ro-bind", system_path, system_path))
        return [*command, "--", executable, *arguments]

    def _invocation_hash(
        self,
        argv: list[str],
        cwd: str,
        timeout_s: float,
        output_cap_bytes: int,
    ) -> str:
        canonical = json.dumps(
            {
                "argv": argv,
                "cwd": cwd,
                "output_cap_bytes": output_cap_bytes,
                "timeout_s": timeout_s,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _audit(
        self,
        invocation_hash: str,
        argv: list[str],
        cwd: str,
        *,
        status: str,
        message: str,
        duration_ms: float,
        job_id: str | None,
    ) -> None:
        entry = {
            "argv": argv,
            "cwd": cwd,
            "duration_ms": round(duration_ms, 3),
            "invocation_hash": invocation_hash,
            "job_id": job_id,
            "message": message,
            "status": status,
            "timestamp": self.wall_clock(),
        }
        with self._audit_lock, self.audit_path.open("a", encoding="utf-8") as audit:
            audit.write(json.dumps(entry, separators=(",", ":"), sort_keys=True))
            audit.write("\n")


class _OutputCollector:
    """Drain both pipes without retaining more than the combined output cap."""

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.stdout = bytearray()
        self.stderr = bytearray()
        self.truncated = False
        self.lock = Lock()

    def drain(self, stream: BinaryIO, destination: Literal["stdout", "stderr"]) -> None:
        target = self.stdout if destination == "stdout" else self.stderr
        while chunk := stream.read(8192):
            with self.lock:
                retained = len(self.stdout) + len(self.stderr)
                remaining = max(0, self.cap - retained)
                target.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    self.truncated = True

    def result(self) -> tuple[bytes, bytes, bool]:
        return bytes(self.stdout), bytes(self.stderr), self.truncated
