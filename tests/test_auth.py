from dataclasses import asdict

import pytest

from ctl.auth import AuthenticationError, JoinAuthManager
from node.auth import sign_join_challenge
from tests.node_doubles import POOL_SECRET, make_profile


class FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class TokenFactory:
    def __init__(self) -> None:
        self.counter = 0

    def __call__(self, _bytes: int) -> str:
        self.counter += 1
        return f"random-value-{self.counter}"


def make_manager(clock: FakeClock) -> JoinAuthManager:
    return JoinAuthManager(
        POOL_SECRET,
        challenge_ttl_s=30.0,
        token_ttl_s=300.0,
        clock=clock,
        token_factory=TokenFactory(),
    )


def complete_join(manager: JoinAuthManager):
    profile = asdict(make_profile())
    challenge = manager.issue_challenge(profile["id"])
    signature = sign_join_challenge(
        POOL_SECRET,
        nonce=challenge.nonce,
        profile=profile,
    )
    return manager.complete_join(profile, challenge.nonce, signature)


def test_valid_challenge_issues_a_short_lived_token() -> None:
    clock = FakeClock()
    manager = make_manager(clock)

    token = complete_join(manager)

    assert token.node_id == "office-01"
    assert token.expires_at == 400.0
    assert manager.validate_token("office-01", token.access_token)


def test_bad_signature_does_not_consume_the_legitimate_challenge() -> None:
    clock = FakeClock()
    manager = make_manager(clock)
    profile = asdict(make_profile())
    challenge = manager.issue_challenge(profile["id"])

    with pytest.raises(AuthenticationError, match="signature"):
        manager.complete_join(profile, challenge.nonce, "0" * 64)

    correct_signature = sign_join_challenge(
        POOL_SECRET,
        nonce=challenge.nonce,
        profile=profile,
    )
    token = manager.complete_join(profile, challenge.nonce, correct_signature)

    assert manager.validate_token(profile["id"], token.access_token)
    with pytest.raises(AuthenticationError, match="already-used"):
        manager.complete_join(profile, challenge.nonce, correct_signature)


def test_second_challenge_does_not_overwrite_a_pending_challenge() -> None:
    clock = FakeClock()
    manager = make_manager(clock)
    profile = asdict(make_profile())
    legitimate = manager.issue_challenge(profile["id"])
    attacker = manager.issue_challenge(profile["id"])

    with pytest.raises(AuthenticationError, match="signature"):
        manager.complete_join(profile, attacker.nonce, "0" * 64)

    legitimate_signature = sign_join_challenge(
        POOL_SECRET,
        nonce=legitimate.nonce,
        profile=profile,
    )
    token = manager.complete_join(
        profile,
        legitimate.nonce,
        legitimate_signature,
    )

    assert manager.validate_token(profile["id"], token.access_token)


def test_expired_challenge_is_rejected() -> None:
    clock = FakeClock()
    manager = make_manager(clock)
    profile = asdict(make_profile())
    challenge = manager.issue_challenge(profile["id"])
    signature = sign_join_challenge(
        POOL_SECRET,
        nonce=challenge.nonce,
        profile=profile,
    )
    clock.now = challenge.expires_at + 0.1

    with pytest.raises(AuthenticationError, match="expired"):
        manager.complete_join(profile, challenge.nonce, signature)


def test_expired_or_wrong_bearer_token_is_rejected() -> None:
    clock = FakeClock()
    manager = make_manager(clock)
    token = complete_join(manager)

    assert not manager.validate_token("office-01", "wrong")
    clock.now = token.expires_at + 0.1
    assert not manager.validate_token("office-01", token.access_token)


def test_revoke_removes_challenges_and_tokens() -> None:
    clock = FakeClock()
    manager = make_manager(clock)
    token = complete_join(manager)

    manager.revoke("office-01")

    assert not manager.validate_token("office-01", token.access_token)
