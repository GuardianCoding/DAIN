from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from node import dain_node
from node.auth import sign_job_request
from node.sandbox import CommandSandbox, SandboxExecutionError, SandboxRejected
from tests.node_doubles import CTL, NODE_ID, POOL_SECRET, make_profile


def sandbox(root: Path, *, allowlist: frozenset[str] | None = None) -> CommandSandbox:
    return CommandSandbox(
        root,
        allowlist=allowlist or frozenset({"cat", "grep", "ls", "sort", "tail"}),
        require_linux_isolation=False,
        bubblewrap=False,
    )


def audit_entries(instance: CommandSandbox) -> list[dict[str, object]]:
    return [json.loads(line) for line in instance.audit_path.read_text().splitlines()]


def signed_exec(
    payload: dict[str, object], *, job_id: str = "job-1"
) -> dict[str, object]:
    issued_at = int(time.time())
    body: dict[str, object] = {
        "job_id": job_id,
        "kind": "exec",
        "payload": payload,
        "shard_index": 0,
        "shard_count": 1,
        "issued_at": issued_at,
    }
    body["signature"] = sign_job_request(
        POOL_SECRET,
        job_id=job_id,
        kind="exec",
        payload=payload,
        shard_index=0,
        shard_count=1,
        issued_at=issued_at,
    )
    return body


def test_executes_allowlisted_command_inside_scratch(tmp_path):
    instance = sandbox(tmp_path)
    (tmp_path / "note.txt").write_text("hello from scratch\n")

    result = instance.execute(["cat", "note.txt"], job_id="job-1")

    assert result.ok is True
    assert result.stdout == "hello from scratch\n"
    assert result.cwd == "."
    assert len(result.invocation_hash) == 64
    assert audit_entries(instance)[0]["status"] == "completed"


@pytest.mark.parametrize(
    "argv",
    [
        ["curl", "https://example.com"],
        ["cat", "~/Documents/private.txt"],
        ["cat", "/etc/passwd"],
        ["cat", "../outside.txt"],
        ["sh", "-c", "cat note.txt"],
        ["sort", "--output=exec-audit.jsonl"],
        ["sort", "-oexec-audit.jsonl"],
        ["grep", "-rR", "secret", "."],
    ],
)
def test_refuses_unsafe_commands_and_paths_and_logs_them(tmp_path, argv):
    instance = sandbox(tmp_path)

    with pytest.raises(SandboxRejected):
        instance.execute(argv, job_id="job-refused")

    entry = audit_entries(instance)[0]
    assert entry["status"] == "refused"
    assert entry["job_id"] == "job-refused"
    assert len(str(entry["invocation_hash"])) == 64


def test_refuses_a_symlink_that_leaves_scratch(tmp_path):
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_text("secret")
    (tmp_path / "escape").symlink_to(outside)
    instance = sandbox(tmp_path)

    with pytest.raises(SandboxRejected, match="links outside scratch"):
        instance.execute(["cat", "escape"])


def test_times_out_and_kills_the_process_group(tmp_path):
    (tmp_path / "empty.txt").touch()
    instance = sandbox(tmp_path)

    result = instance.execute(["tail", "-f", "empty.txt"], timeout_s=0.05)

    assert result.ok is False
    assert result.timed_out is True
    assert result.exit_code is None
    assert audit_entries(instance)[0]["status"] == "timed_out"


def test_caps_combined_output(tmp_path):
    (tmp_path / "large.txt").write_text("x" * 200)
    instance = sandbox(tmp_path)

    result = instance.execute(["cat", "large.txt"], output_cap_bytes=32)

    assert result.truncated is True
    assert len(result.stdout.encode()) + len(result.stderr.encode()) == 32


def test_working_directory_is_jailed(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "inside.txt").write_text("inside")
    instance = sandbox(tmp_path)

    result = instance.execute(["cat", "inside.txt"], cwd="nested")

    assert result.ok is True
    assert result.stdout == "inside"
    assert result.cwd == "nested"


def test_linux_isolation_is_fail_closed_without_bubblewrap(tmp_path):
    instance = CommandSandbox(
        tmp_path,
        allowlist=frozenset({"ls"}),
        require_linux_isolation=True,
        bubblewrap=False,
    )

    with pytest.raises(SandboxRejected, match="bubblewrap is required"):
        instance.execute(["ls"])

    assert audit_entries(instance)[0]["status"] == "refused"


def test_isolation_is_required_by_default_on_every_platform(tmp_path):
    instance = CommandSandbox(
        tmp_path,
        allowlist=frozenset({"ls"}),
        bubblewrap=False,
    )

    with pytest.raises(SandboxRejected, match="bubblewrap is required"):
        instance.execute(["ls"])


def test_isolation_runtime_failure_is_logged(tmp_path):
    instance = CommandSandbox(
        tmp_path,
        allowlist=frozenset({"ls"}),
        require_linux_isolation=True,
        bubblewrap="/does/not/exist/bwrap",
    )

    with pytest.raises(SandboxExecutionError, match="could not start"):
        instance.execute(["ls"], job_id="job-runtime")

    entry = audit_entries(instance)[0]
    assert entry["status"] == "failed"
    assert entry["job_id"] == "job-runtime"


def test_exec_endpoint_requires_a_valid_job_signature(tmp_path):
    agent = dain_node.configure(
        make_profile(),
        ctl=CTL,
        pool_secret=POOL_SECRET,
        sandbox=sandbox(tmp_path),
    )
    client = TestClient(dain_node.app)
    request = signed_exec({"argv": ["ls"]})
    request["signature"] = "0" * 64

    response = client.post("/exec", json=request)

    assert response.status_code == 403
    assert not agent.sandbox.audit_path.exists()


def test_exec_endpoint_returns_result_and_audits_job(tmp_path):
    agent = dain_node.configure(
        make_profile(),
        ctl=CTL,
        pool_secret=POOL_SECRET,
        sandbox=sandbox(tmp_path),
    )
    client = TestClient(dain_node.app)

    response = client.post("/exec", json=signed_exec({"argv": ["ls"]}))

    assert response.status_code == 200
    assert response.json()["result"]["node_id"] == NODE_ID
    assert response.json()["result"]["ok"] is True
    assert audit_entries(agent.sandbox)[0]["job_id"] == "job-1"


def test_exec_endpoint_exposes_logged_policy_refusal(tmp_path):
    agent = dain_node.configure(
        make_profile(),
        ctl=CTL,
        pool_secret=POOL_SECRET,
        sandbox=sandbox(tmp_path),
    )
    client = TestClient(dain_node.app)

    response = client.post(
        "/exec", json=signed_exec({"argv": ["curl", "https://example.com"]})
    )

    assert response.status_code == 422
    assert "not allowlisted" in response.json()["detail"]
    assert audit_entries(agent.sandbox)[0]["status"] == "refused"
