"""Control-plane state for nonce-based joins and short-lived node tokens."""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Any

from node.auth import verify_join_challenge


class AuthenticationError(RuntimeError):
    pass


@dataclass(frozen=True)
class JoinChallenge:
    node_id: str
    nonce: str
    expires_at: float


@dataclass(frozen=True)
class NodeToken:
    node_id: str
    access_token: str
    expires_at: float


class JoinAuthManager:
    def __init__(
        self,
        pool_secret: str,
        *,
        challenge_ttl_s: float = 30.0,
        token_ttl_s: float = 300.0,
        clock: Callable[[], float] = time.time,
        token_factory: Callable[[int], str] = secrets.token_urlsafe,
    ) -> None:
        if not pool_secret:
            raise ValueError("pool_secret must not be empty")
        if challenge_ttl_s <= 0:
            raise ValueError("challenge_ttl_s must be greater than zero")
        if token_ttl_s <= 0:
            raise ValueError("token_ttl_s must be greater than zero")

        self.pool_secret = pool_secret
        self.challenge_ttl_s = challenge_ttl_s
        self.token_ttl_s = token_ttl_s
        self.clock = clock
        self.token_factory = token_factory
        self.challenges: dict[str, JoinChallenge] = {}
        self.tokens: dict[str, NodeToken] = {}
        self.lock = RLock()

    def issue_challenge(self, node_id: str) -> JoinChallenge:
        if not node_id.strip():
            raise ValueError("node_id must not be empty")

        with self.lock:
            now = self.clock()
            self._prune_expired_challenges(now)
            challenge = JoinChallenge(
                node_id=node_id,
                nonce=self.token_factory(32),
                expires_at=now + self.challenge_ttl_s,
            )
            self.challenges[challenge.nonce] = challenge
            return challenge

    def complete_join(
        self,
        profile: Mapping[str, Any],
        nonce: str,
        signature: str,
    ) -> NodeToken:
        node_id = profile.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise AuthenticationError("profile has no node id")

        with self.lock:
            now = self.clock()
            challenge = self.challenges.get(nonce)

            if challenge is None or challenge.node_id != node_id:
                raise AuthenticationError("invalid or already-used join challenge")
            if now > challenge.expires_at:
                self.challenges.pop(nonce, None)
                raise AuthenticationError("join challenge expired")
            if not verify_join_challenge(
                self.pool_secret,
                nonce=nonce,
                profile=profile,
                signature=signature,
            ):
                raise AuthenticationError("invalid join signature")

            self.challenges.pop(nonce, None)
            token = NodeToken(
                node_id=node_id,
                access_token=self.token_factory(32),
                expires_at=now + self.token_ttl_s,
            )
            self.tokens[node_id] = token
            return token

    def validate_token(self, node_id: str, access_token: str) -> bool:
        with self.lock:
            token = self.tokens.get(node_id)
            if token is None:
                return False
            if self.clock() > token.expires_at:
                self.tokens.pop(node_id, None)
                return False
            return secrets.compare_digest(token.access_token, access_token)

    def revoke(self, node_id: str) -> None:
        with self.lock:
            self.challenges = {
                nonce: challenge
                for nonce, challenge in self.challenges.items()
                if challenge.node_id != node_id
            }
            self.tokens.pop(node_id, None)

    def reset(self) -> None:
        with self.lock:
            self.challenges.clear()
            self.tokens.clear()

    def _prune_expired_challenges(self, now: float) -> None:
        self.challenges = {
            nonce: challenge
            for nonce, challenge in self.challenges.items()
            if challenge.expires_at >= now
        }
