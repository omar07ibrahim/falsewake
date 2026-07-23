"""Fixed-profile Experiment 003 facade over the shared run authority.

The capability type, process-local registry, issuance latch, and verification
engine live only in :mod:`falsewake.experiment_002_run_authority`.  This module
does not accept a profile, path, digest, or namespace from its callers.  Every
route below binds the retained capability to profile ``003`` before delegating.
"""

from __future__ import annotations

from falsewake import experiment_002_run_authority as _engine

VerifiedRunRegistration = _engine.VerifiedRunRegistration
Experiment003RunAuthorityError = _engine.Experiment002RunAuthorityError
_RegisteredChildInputSnapshot = _engine._RegisteredChildInputSnapshot
_VerifiedState = _engine._VerifiedState

_MAX_CHILD_BUNDLE_BYTES = _engine._MAX_CHILD_BUNDLE_BYTES
_SEALED_CHILD_BUNDLE_FD = _engine._SEALED_CHILD_BUNDLE_FD
_SEALED_CHILD_MEMFD_TARGET = _engine._EXPERIMENT_003_SEALED_CHILD_MEMFD_TARGET

__all__ = (
    "Experiment003RunAuthorityError",
    "VerifiedRunRegistration",
    "reverify_verified_run_registration",
    "verify_and_issue_experiment_003_run_registration",
    "verify_verified_run_registration",
)


def verify_and_issue_experiment_003_run_registration() -> VerifiedRunRegistration:
    """Verify the fixed registration and mint the one profile-003 capability."""

    return _engine._verify_and_issue_experiment_003_run_registration()


def _verify_and_issue_experiment_003_sealed_child_registration() -> (
    VerifiedRunRegistration
):
    """Mint a profile-003 capability from the fixed sealed child bundle."""

    return _engine._verify_and_issue_experiment_003_sealed_child_registration()


def verify_verified_run_registration(
    registration: VerifiedRunRegistration,
) -> None:
    """Fully verify an exact profile-003 capability."""

    _engine._verify_experiment_003_run_registration(registration)


def reverify_verified_run_registration(
    registration: VerifiedRunRegistration,
) -> None:
    """Reverify an exact profile-003 capability from its retained state."""

    _engine._reverify_experiment_003_run_registration(registration)


def _registered_child_input_snapshot(
    registration: VerifiedRunRegistration,
    /,
) -> _RegisteredChildInputSnapshot:
    """Return retained child inputs only for a sealed profile-003 capability."""

    _engine._require_verified_registration_profile(
        registration,
        _engine._EXPERIMENT_003_PROFILE,
    )
    return _engine._registered_child_input_snapshot(registration)


def _create_sealed_experiment_003_child_bundle_fd(
    registration: VerifiedRunRegistration,
    /,
) -> int:
    """Create the fixed profile-003 sealed child source-bundle descriptor."""

    return _engine._create_sealed_experiment_003_child_bundle_fd(registration)


def _verified_state(
    registration: VerifiedRunRegistration,
    /,
) -> _VerifiedState:
    """Return shared retained state only when its exact profile is ``003``."""

    return _engine._require_verified_registration_profile(
        registration,
        _engine._EXPERIMENT_003_PROFILE,
    )


def _required_child_bundle_seals() -> int:
    """Return the shared kernel's exact required memfd seal mask."""

    return _engine._required_child_bundle_seals()
