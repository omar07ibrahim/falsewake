from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import falsewake.experiment_002_training_bridge as bridge

_DIGEST = "0" * 64
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _run_isolated_bridge_probe(source: str) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(_REPOSITORY_ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=_REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _chain_digest(kind: str, generation: int) -> str:
    return hashlib.sha256(f"{kind}:{generation}".encode()).hexdigest()


def _registered_transition(
    authority: bridge.RegisteredExecutorSessionAuthority,
    *,
    epoch_token: object,
    batch_token: object,
    global_update: int,
) -> bridge.RegisteredOptimizerTransition:
    return bridge._issue_registered_optimizer_transition(
        executor_authority=authority,
        epoch_session_token=epoch_token,
        batch_token=batch_token,
        zero_based_epoch=global_update // bridge._UPDATES_PER_EPOCH,
        zero_based_global_update=global_update,
        batch_size=bridge._REGISTERED_BATCH_SIZE,
        learning_rate=bridge._registered_learning_rate(global_update),
        batch_mean_training_loss=np.float32(0.5),
        returned_preclip_l2_norm=np.float32(1.0),
        optimizer_generation_before=global_update,
        optimizer_generation_after=global_update + 1,
        model_sha256_before=(
            _DIGEST if global_update == 0 else _chain_digest("model", global_update - 1)
        ),
        model_sha256_after=_chain_digest("model", global_update),
        optimizer_sha256_before=(
            "2" * 64
            if global_update == 0
            else _chain_digest("optimizer", global_update - 1)
        ),
        optimizer_sha256_after=_chain_digest("optimizer", global_update),
        rng_sha256_before=(
            "4" * 64 if global_update == 0 else _chain_digest("rng", global_update - 1)
        ),
        rng_sha256_after=_chain_digest("rng", global_update),
    )


def _accept(
    authority: bridge.RegisteredExecutorSessionAuthority,
    transition: bridge.RegisteredOptimizerTransition,
    *,
    epoch_token: object,
    batch_token: object,
    receipt_token: object,
) -> bridge.TraceConsumedTransition:
    snapshot = bridge._transition_snapshot(transition, required_phase="ISSUED")
    consumption = bridge._mark_transition_trace_consumed(
        transition,
        receipt_token=receipt_token,
        trace_session_token=object(),
        trace_record_index=snapshot.zero_based_global_update,
    )
    bridge._accept_trace_consumed_transition(
        transition,
        consumption,
        receipt_token=receipt_token,
        executor_authority=authority,
        epoch_session_token=epoch_token,
        batch_token=batch_token,
    )
    return consumption


def test_session_generation_ledger_rejects_coherent_replay_and_forward_skip() -> None:
    _run_isolated_bridge_probe(
        """
        import numpy as np
        import falsewake.experiment_002_training_bridge as bridge

        def issue(authority, update, model_before, model_after,
                  optimizer_before, optimizer_after, rng_before, rng_after):
            epoch_token = object()
            batch_token = object()
            transition = bridge._issue_registered_optimizer_transition(
                executor_authority=authority,
                epoch_session_token=epoch_token,
                batch_token=batch_token,
                zero_based_epoch=update // bridge._UPDATES_PER_EPOCH,
                zero_based_global_update=update,
                batch_size=bridge._REGISTERED_BATCH_SIZE,
                learning_rate=bridge._registered_learning_rate(update),
                batch_mean_training_loss=np.float32(0.5),
                returned_preclip_l2_norm=np.float32(1.0),
                optimizer_generation_before=update,
                optimizer_generation_after=update + 1,
                model_sha256_before=model_before,
                model_sha256_after=model_after,
                optimizer_sha256_before=optimizer_before,
                optimizer_sha256_after=optimizer_after,
                rng_sha256_before=rng_before,
                rng_sha256_after=rng_after,
            )
            return transition, epoch_token, batch_token

        def accept(authority, transition, epoch_token, batch_token):
            receipt_token = object()
            consumption = bridge._mark_transition_trace_consumed(
                transition,
                receipt_token=receipt_token,
                trace_session_token=object(),
                trace_record_index=0,
            )
            bridge._accept_trace_consumed_transition(
                transition,
                consumption,
                receipt_token=receipt_token,
                executor_authority=authority,
                epoch_session_token=epoch_token,
                batch_token=batch_token,
            )

        first = bridge._issue_registered_executor_session(20_260_720)
        transition, epoch_token, batch_token = issue(
            first, 0, "0" * 64, "1" * 64, "2" * 64, "3" * 64,
            "4" * 64, "5" * 64,
        )
        accept(first, transition, epoch_token, batch_token)
        original_ledger = bridge._EXECUTOR_GENERATION_LEDGERS[first]
        original_anchor = bridge._EXECUTOR_GENERATION_ANCHORS[first]
        original_history_size = len(bridge._EXECUTOR_GENERATION_HISTORY)
        try:
            original_anchor.history.clear()
        except AttributeError:
            pass
        else:
            raise AssertionError("generation anchor exposed a clearable history")
        try:
            original_anchor.__init__(first)
        except TypeError:
            pass
        else:
            raise AssertionError("generation anchor allowed reinitialization")
        try:
            del original_anchor._latest_generation
        except TypeError:
            pass
        else:
            raise AssertionError("generation anchor allowed head deletion")
        state = bridge._EXECUTOR_SESSIONS[first]
        state.next_optimizer_generation = 0
        bridge._EXECUTOR_SESSION_LIFECYCLES[first] = (
            bridge._ExecutorSessionLifecycleGuard(0, None, False)
        )
        bridge._EXECUTOR_GENERATION_LEDGERS[first] = ()
        bridge._EXECUTOR_GENERATION_ANCHORS[first] = (
            bridge._ExecutorGenerationAnchor(first, 20_260_720)
        )
        try:
            issue(
                first, 0, "0" * 64, "1" * 64, "2" * 64, "3" * 64,
                "4" * 64, "5" * 64,
            )
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("accepted generation was replayed")
        assert state.failed
        assert first in bridge._FAILED_EXECUTOR_SESSIONS
        assert sum(
            item.executor_authority is first
            for item in bridge._EXECUTOR_GENERATION_HISTORY
        ) == 1
        assert len(bridge._EXECUTOR_GENERATION_HISTORY) == original_history_size

        # Restore every replaceable mirror; raw terminal state and the
        # append-only generation authority still prevent continuation.
        state.next_optimizer_generation = 1
        bridge._EXECUTOR_GENERATION_LEDGERS[first] = original_ledger
        bridge._EXECUTOR_GENERATION_ANCHORS[first] = original_anchor
        bridge._EXECUTOR_SESSION_LIFECYCLES[first] = (
            bridge._ExecutorSessionLifecycleGuard(1, None, True)
        )
        try:
            issue(
                first, 1, "1" * 64, "6" * 64, "3" * 64, "7" * 64,
                "5" * 64, "8" * 64,
            )
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("terminal replay session continued")

        second = bridge._issue_registered_executor_session(20_260_721)
        second_state = bridge._EXECUTOR_SESSIONS[second]
        second_state.next_optimizer_generation = bridge._UPDATES_PER_EPOCH
        bridge._EXECUTOR_SESSION_LIFECYCLES[second] = (
            bridge._ExecutorSessionLifecycleGuard(
                bridge._UPDATES_PER_EPOCH, None, False
            )
        )
        try:
            bridge._require_registered_executor_generation(
                second, bridge._UPDATES_PER_EPOCH
            )
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("forward-skipped epoch began")
        assert second_state.failed

        third = bridge._issue_registered_executor_session(20_260_719)
        try:
            issue(
                third, bridge._UPDATES_PER_EPOCH,
                "0" * 64, "1" * 64, "2" * 64, "3" * 64,
                "4" * 64, "5" * 64,
            )
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("forward-skipped transition was issued")
        assert bridge._EXECUTOR_SESSIONS[third].failed
        """
    )


def test_session_state_digest_chain_rejects_discontinuity_terminally() -> None:
    _run_isolated_bridge_probe(
        """
        import numpy as np
        import falsewake.experiment_002_training_bridge as bridge

        authority = bridge._issue_registered_executor_session(20_260_720)
        epoch_token = object()
        first_batch_token = object()

        def issue(update, model_before, model_after, optimizer_before,
                  optimizer_after, rng_before, rng_after, batch_token):
            return bridge._issue_registered_optimizer_transition(
                executor_authority=authority,
                epoch_session_token=epoch_token,
                batch_token=batch_token,
                zero_based_epoch=0,
                zero_based_global_update=update,
                batch_size=bridge._REGISTERED_BATCH_SIZE,
                learning_rate=bridge._registered_learning_rate(update),
                batch_mean_training_loss=np.float32(0.5),
                returned_preclip_l2_norm=np.float32(1.0),
                optimizer_generation_before=update,
                optimizer_generation_after=update + 1,
                model_sha256_before=model_before,
                model_sha256_after=model_after,
                optimizer_sha256_before=optimizer_before,
                optimizer_sha256_after=optimizer_after,
                rng_sha256_before=rng_before,
                rng_sha256_after=rng_after,
            )

        first = issue(
            0, "0" * 64, "1" * 64, "2" * 64, "3" * 64,
            "4" * 64, "5" * 64, first_batch_token,
        )
        receipt_token = object()
        consumption = bridge._mark_transition_trace_consumed(
            first,
            receipt_token=receipt_token,
            trace_session_token=object(),
            trace_record_index=0,
        )
        bridge._accept_trace_consumed_transition(
            first,
            consumption,
            receipt_token=receipt_token,
            executor_authority=authority,
            epoch_session_token=epoch_token,
            batch_token=first_batch_token,
        )

        try:
            issue(
                1, "a" * 64, "6" * 64, "3" * 64, "7" * 64,
                "5" * 64, "8" * 64, object(),
            )
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("discontinuous digest chain was issued")
        state = bridge._EXECUTOR_SESSIONS[authority]
        assert state.failed
        assert authority in bridge._FAILED_EXECUTOR_SESSIONS
        assert len(bridge._EXECUTOR_GENERATION_LEDGERS[authority]) == 1
        assert sum(
            item.executor_authority is authority
            for item in bridge._EXECUTOR_GENERATION_HISTORY
        ) == 1
        try:
            issue(
                1, "1" * 64, "6" * 64, "3" * 64, "7" * 64,
                "5" * 64, "8" * 64, object(),
            )
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("terminal digest session retried")
        """
    )


def test_registered_session_seed_guard_survives_mutable_set_reset() -> None:
    _run_isolated_bridge_probe(
        """
        import falsewake.experiment_002_training_bridge as bridge

        authority = bridge._issue_registered_executor_session(20_260_720)
        counts = (
            len(bridge._EXECUTOR_SESSIONS),
            len(bridge._EXECUTOR_GENERATION_ANCHORS),
            len(bridge._EXECUTOR_GENERATION_ANCHOR_HISTORY),
            len(bridge._ISSUED_EXECUTOR_GENERATION_ANCHORS),
        )
        bridge._ISSUED_REGISTERED_SEEDS.clear()
        try:
            bridge._issue_registered_executor_session(20_260_720)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("duplicate same-seed session was issued")
        assert counts == (
            len(bridge._EXECUTOR_SESSIONS),
            len(bridge._EXECUTOR_GENERATION_ANCHORS),
            len(bridge._EXECUTOR_GENERATION_ANCHOR_HISTORY),
            len(bridge._ISSUED_EXECUTOR_GENERATION_ANCHORS),
        )
        assert bridge._EXECUTOR_GENERATION_ANCHOR_HISTORY[0].seed == 20_260_720
        assert (
            bridge._EXECUTOR_GENERATION_ANCHOR_HISTORY[0].executor_authority
            is authority
        )
        assert not bridge._EXECUTOR_SESSIONS[authority].failed
        """
    )


def test_session_snapshot_terminalizes_anchor_authority_corruption() -> None:
    _run_isolated_bridge_probe(
        """
        import falsewake.experiment_002_training_bridge as bridge

        authority = bridge._issue_registered_executor_session(20_260_720)
        anchor = bridge._EXECUTOR_GENERATION_ANCHORS[authority]
        bridge._EXECUTOR_GENERATION_ANCHOR_HISTORY.clear()
        bridge._ISSUED_EXECUTOR_GENERATION_ANCHORS.clear()
        try:
            bridge._executor_session_snapshot(authority)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("corrupt anchor authority remained readable")
        state = bridge._EXECUTOR_SESSIONS[authority]
        assert state.failed
        assert state.active_transition_token is None
        assert authority in bridge._FAILED_EXECUTOR_SESSIONS
        assert bridge._EXECUTOR_SESSION_LIFECYCLES[authority].failed

        bridge._EXECUTOR_GENERATION_ANCHOR_HISTORY.append(anchor)
        bridge._ISSUED_EXECUTOR_GENERATION_ANCHORS.add(anchor)
        try:
            bridge._executor_session_snapshot(authority)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("terminal anchor session resumed after restoration")
        try:
            bridge._require_registered_executor_generation(authority, 0)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("terminal anchor session began an epoch")
        """
    )


def test_direct_validator_corruption_is_terminal_before_restore_retry() -> None:
    _run_isolated_bridge_probe(
        """
        import numpy as np
        import falsewake.experiment_002_training_bridge as bridge

        def issue(authority):
            epoch_token = object()
            batch_token = object()
            transition = bridge._issue_registered_optimizer_transition(
                executor_authority=authority,
                epoch_session_token=epoch_token,
                batch_token=batch_token,
                zero_based_epoch=0,
                zero_based_global_update=0,
                batch_size=bridge._REGISTERED_BATCH_SIZE,
                learning_rate=bridge._registered_learning_rate(0),
                batch_mean_training_loss=np.float32(0.5),
                returned_preclip_l2_norm=np.float32(1.0),
                optimizer_generation_before=0,
                optimizer_generation_after=1,
                model_sha256_before="0" * 64,
                model_sha256_after="1" * 64,
                optimizer_sha256_before="2" * 64,
                optimizer_sha256_after="3" * 64,
                rng_sha256_before="4" * 64,
                rng_sha256_after="5" * 64,
            )
            return transition, epoch_token, batch_token

        first = bridge._issue_registered_executor_session(20_260_720)
        first_transition, _, _ = issue(first)
        saved_phase = bridge._TRANSITION_PHASES.pop(first_transition)
        try:
            bridge._transition_snapshot(first_transition)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("missing transition phase guard was ignored")
        assert bridge._ISSUED_TRANSITIONS[first_transition].phase == "FAILED"
        assert bridge._EXECUTOR_SESSIONS[first].failed
        assert first_transition in bridge._INVALID_TRANSITIONS
        bridge._TRANSITION_PHASES[first_transition] = saved_phase
        try:
            bridge._transition_snapshot(first_transition)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("invalid transition resumed after mirror restore")

        second = bridge._issue_registered_executor_session(20_260_721)
        second_transition, _, _ = issue(second)
        receipt_token = object()
        consumption = bridge._mark_transition_trace_consumed(
            second_transition,
            receipt_token=receipt_token,
            trace_session_token=object(),
            trace_record_index=0,
        )
        saved_lifecycle = bridge._CONSUMPTION_LIFECYCLES.pop(consumption)
        try:
            bridge._trace_consumption_snapshot(consumption)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("missing consumption lifecycle was ignored")
        consumption_state = bridge._ISSUED_CONSUMPTIONS[consumption]
        assert consumption_state.failed and not consumption_state.used
        assert bridge._ISSUED_TRANSITIONS[second_transition].phase == "FAILED"
        assert bridge._EXECUTOR_SESSIONS[second].failed
        assert consumption in bridge._INVALID_CONSUMPTIONS
        bridge._CONSUMPTION_LIFECYCLES[consumption] = saved_lifecycle
        try:
            bridge._trace_consumption_snapshot(consumption)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("invalid consumption resumed after mirror restore")
        """
    )


def test_accepted_validator_corruption_poison_preserves_committed_winner() -> None:
    _run_isolated_bridge_probe(
        """
        import numpy as np
        import falsewake.experiment_002_training_bridge as bridge

        def accepted(authority):
            epoch_token = object()
            batch_token = object()
            receipt_token = object()
            transition = bridge._issue_registered_optimizer_transition(
                executor_authority=authority,
                epoch_session_token=epoch_token,
                batch_token=batch_token,
                zero_based_epoch=0,
                zero_based_global_update=0,
                batch_size=bridge._REGISTERED_BATCH_SIZE,
                learning_rate=bridge._registered_learning_rate(0),
                batch_mean_training_loss=np.float32(0.5),
                returned_preclip_l2_norm=np.float32(1.0),
                optimizer_generation_before=0,
                optimizer_generation_after=1,
                model_sha256_before="0" * 64,
                model_sha256_after="1" * 64,
                optimizer_sha256_before="2" * 64,
                optimizer_sha256_after="3" * 64,
                rng_sha256_before="4" * 64,
                rng_sha256_after="5" * 64,
            )
            consumption = bridge._mark_transition_trace_consumed(
                transition,
                receipt_token=receipt_token,
                trace_session_token=object(),
                trace_record_index=0,
            )
            bridge._accept_trace_consumed_transition(
                transition,
                consumption,
                receipt_token=receipt_token,
                executor_authority=authority,
                epoch_session_token=epoch_token,
                batch_token=batch_token,
            )
            return transition, consumption

        first = bridge._issue_registered_executor_session(20_260_720)
        first_transition, first_consumption = accepted(first)
        saved_phase = bridge._TRANSITION_PHASES.pop(first_transition)
        try:
            bridge._transition_snapshot(first_transition)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("accepted phase corruption was ignored")
        assert (
            bridge._ISSUED_TRANSITIONS[first_transition].phase
            == "POPULATION_ACCEPTED"
        )
        assert bridge._ISSUED_CONSUMPTIONS[first_consumption].used
        assert not bridge._ISSUED_CONSUMPTIONS[first_consumption].failed
        assert first_transition in bridge._ACCEPTED_TRANSITIONS
        assert first_transition in bridge._INVALID_TRANSITIONS
        assert bridge._EXECUTOR_SESSIONS[first].failed
        bridge._TRANSITION_PHASES[first_transition] = saved_phase
        try:
            bridge._transition_snapshot(first_transition)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("accepted invalid transition resumed")

        second = bridge._issue_registered_executor_session(20_260_721)
        second_transition, second_consumption = accepted(second)
        saved_lifecycle = bridge._CONSUMPTION_LIFECYCLES.pop(second_consumption)
        try:
            bridge._trace_consumption_snapshot(second_consumption)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("used consumption corruption was ignored")
        second_state = bridge._ISSUED_CONSUMPTIONS[second_consumption]
        assert second_state.used and not second_state.failed
        assert (
            bridge._ISSUED_TRANSITIONS[second_transition].phase
            == "POPULATION_ACCEPTED"
        )
        assert second_transition in bridge._ACCEPTED_TRANSITIONS
        assert second_consumption in bridge._INVALID_CONSUMPTIONS
        assert bridge._EXECUTOR_SESSIONS[second].failed
        bridge._CONSUMPTION_LIFECYCLES[second_consumption] = saved_lifecycle
        try:
            bridge._trace_consumption_snapshot(second_consumption)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("used invalid consumption resumed")
        """
    )


@pytest.mark.parametrize(
    "missing_guard",
    (
        "transition_guard",
        "consumption_guard",
        "transition_consumption",
        "transition_claim",
        "generation_history",
        "generation_set",
        "session_guard",
    ),
)
def test_accepted_single_guard_loss_terminalizes_owner_by_consensus(
    missing_guard: str,
) -> None:
    _run_isolated_bridge_probe(
        f"""
        import numpy as np
        import falsewake.experiment_002_training_bridge as bridge

        authority = bridge._issue_registered_executor_session(20_260_720)
        epoch_token = object()
        batch_token = object()

        def issue(update):
            return bridge._issue_registered_optimizer_transition(
                executor_authority=authority,
                epoch_session_token=epoch_token,
                batch_token=object() if update else batch_token,
                zero_based_epoch=0,
                zero_based_global_update=update,
                batch_size=bridge._REGISTERED_BATCH_SIZE,
                learning_rate=bridge._registered_learning_rate(update),
                batch_mean_training_loss=np.float32(0.5),
                returned_preclip_l2_norm=np.float32(1.0),
                optimizer_generation_before=update,
                optimizer_generation_after=update + 1,
                model_sha256_before="0" * 64 if update == 0 else "1" * 64,
                model_sha256_after="1" * 64 if update == 0 else "6" * 64,
                optimizer_sha256_before="2" * 64 if update == 0 else "3" * 64,
                optimizer_sha256_after="3" * 64 if update == 0 else "7" * 64,
                rng_sha256_before="4" * 64 if update == 0 else "5" * 64,
                rng_sha256_after="5" * 64 if update == 0 else "8" * 64,
            )

        transition = issue(0)
        receipt_token = object()
        consumption = bridge._mark_transition_trace_consumed(
            transition,
            receipt_token=receipt_token,
            trace_session_token=object(),
            trace_record_index=0,
        )
        bridge._accept_trace_consumed_transition(
            transition,
            consumption,
            receipt_token=receipt_token,
            executor_authority=authority,
            epoch_session_token=epoch_token,
            batch_token=batch_token,
        )

        missing_guard = {missing_guard!r}
        saved = None
        if missing_guard == "transition_guard":
            saved = bridge._TRANSITION_GUARDS.pop(transition)
        elif missing_guard == "consumption_guard":
            saved = bridge._CONSUMPTION_GUARDS.pop(consumption)
        elif missing_guard == "transition_consumption":
            saved = bridge._TRANSITION_CONSUMPTIONS.pop(transition)
        elif missing_guard == "transition_claim":
            saved = bridge._TRANSITION_CLAIM_GUARDS.pop(transition)
        elif missing_guard == "generation_history":
            saved = tuple(bridge._EXECUTOR_GENERATION_HISTORY)
            bridge._EXECUTOR_GENERATION_HISTORY.clear()
        elif missing_guard == "generation_set":
            saved = set(bridge._ISSUED_EXECUTOR_GENERATIONS)
            bridge._ISSUED_EXECUTOR_GENERATIONS.clear()
        elif missing_guard == "session_guard":
            saved = bridge._EXECUTOR_SESSION_GUARDS.pop(authority)
        else:
            raise AssertionError("unknown missing guard")

        try:
            if missing_guard in {{
                "consumption_guard",
                "transition_consumption",
                "transition_claim",
            }}:
                bridge._trace_consumption_snapshot(consumption)
            else:
                bridge._transition_snapshot(transition)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("single guard loss was ignored")

        assert bridge._ISSUED_TRANSITIONS[transition].phase == "POPULATION_ACCEPTED"
        assert bridge._ISSUED_TRANSITIONS[transition].invalid
        assert bridge._ISSUED_CONSUMPTIONS[consumption].used
        assert not bridge._ISSUED_CONSUMPTIONS[consumption].failed
        assert bridge._ISSUED_CONSUMPTIONS[consumption].invalid
        assert bridge._EXECUTOR_SESSIONS[authority].failed

        if missing_guard == "transition_guard":
            bridge._TRANSITION_GUARDS[transition] = saved
        elif missing_guard == "consumption_guard":
            bridge._CONSUMPTION_GUARDS[consumption] = saved
        elif missing_guard == "transition_consumption":
            bridge._TRANSITION_CONSUMPTIONS[transition] = saved
        elif missing_guard == "transition_claim":
            bridge._TRANSITION_CLAIM_GUARDS[transition] = saved
        elif missing_guard == "generation_history":
            bridge._EXECUTOR_GENERATION_HISTORY.extend(saved)
        elif missing_guard == "generation_set":
            bridge._ISSUED_EXECUTOR_GENERATIONS.update(saved)
        elif missing_guard == "session_guard":
            bridge._EXECUTOR_SESSION_GUARDS[authority] = saved

        bridge._INVALID_TRANSITIONS.clear()
        bridge._INVALID_CONSUMPTIONS.clear()
        try:
            bridge._trace_consumption_snapshot(consumption)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("raw invalid poison was reset with weak sets")
        try:
            issue(1)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("terminal owner issued the next generation")
        assert len(bridge._EXECUTOR_GENERATION_LEDGERS[authority]) == 1
        """
    )


@pytest.mark.parametrize("corrupted_capability", ("transition", "consumption"))
def test_consensus_poison_never_terminalizes_lower_support_foreign_owner(
    corrupted_capability: str,
) -> None:
    _run_isolated_bridge_probe(
        f"""
        from dataclasses import replace
        import numpy as np
        import falsewake.experiment_002_training_bridge as bridge

        def issue(authority, epoch_token, batch_token):
            return bridge._issue_registered_optimizer_transition(
                executor_authority=authority,
                epoch_session_token=epoch_token,
                batch_token=batch_token,
                zero_based_epoch=0,
                zero_based_global_update=0,
                batch_size=bridge._REGISTERED_BATCH_SIZE,
                learning_rate=bridge._registered_learning_rate(0),
                batch_mean_training_loss=np.float32(0.5),
                returned_preclip_l2_norm=np.float32(1.0),
                optimizer_generation_before=0,
                optimizer_generation_after=1,
                model_sha256_before="0" * 64,
                model_sha256_after="1" * 64,
                optimizer_sha256_before="2" * 64,
                optimizer_sha256_after="3" * 64,
                rng_sha256_before="4" * 64,
                rng_sha256_after="5" * 64,
            )

        local = bridge._issue_registered_executor_session(20_260_720)
        local_epoch = object()
        local_batch = object()
        local_transition = issue(local, local_epoch, local_batch)
        receipt_token = object()
        local_consumption = bridge._mark_transition_trace_consumed(
            local_transition,
            receipt_token=receipt_token,
            trace_session_token=object(),
            trace_record_index=0,
        )
        bridge._accept_trace_consumed_transition(
            local_transition,
            local_consumption,
            receipt_token=receipt_token,
            executor_authority=local,
            epoch_session_token=local_epoch,
            batch_token=local_batch,
        )

        foreign = bridge._issue_registered_executor_session(20_260_721)
        foreign_transition = issue(foreign, object(), object())
        corruption = {corrupted_capability!r}
        if corruption == "transition":
            foreign_token = bridge._EXECUTOR_SESSIONS[foreign].issuer.session_token
            local_state = bridge._ISSUED_TRANSITIONS[local_transition]
            local_state.issuer = replace(
                local_state.issuer,
                executor_authority=foreign,
                executor_session_token=foreign_token,
            )
            bridge._TRANSITION_GUARDS[local_transition] = replace(
                bridge._TRANSITION_GUARDS[local_transition],
                executor_authority=foreign,
                executor_session_token=foreign_token,
            )
            accessor = lambda: bridge._transition_snapshot(local_transition)
        else:
            consumption_state = bridge._ISSUED_CONSUMPTIONS[local_consumption]
            consumption_state.issuer = replace(
                consumption_state.issuer,
                transition=foreign_transition,
            )
            bridge._CONSUMPTION_GUARDS[local_consumption] = replace(
                bridge._CONSUMPTION_GUARDS[local_consumption],
                transition=foreign_transition,
            )
            accessor = lambda: bridge._trace_consumption_snapshot(local_consumption)
        try:
            accessor()
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("coherent lower-support corruption was ignored")
        assert bridge._EXECUTOR_SESSIONS[local].failed
        assert bridge._ISSUED_TRANSITIONS[local_transition].invalid
        assert bridge._ISSUED_TRANSITIONS[foreign_transition].phase == "ISSUED"
        assert not bridge._EXECUTOR_SESSIONS[foreign].failed
        bridge._transition_snapshot(foreign_transition, required_phase="ISSUED")
        bridge._executor_session_snapshot(foreign)
        """
    )


def test_bound_consumption_consensus_prefers_local_issuer_over_two_foreign_links() -> (
    None
):
    _run_isolated_bridge_probe(
        """
        import numpy as np
        import falsewake.experiment_002_training_bridge as bridge

        def trace_consumed(authority):
            transition = bridge._issue_registered_optimizer_transition(
                executor_authority=authority,
                epoch_session_token=object(),
                batch_token=object(),
                zero_based_epoch=0,
                zero_based_global_update=0,
                batch_size=bridge._REGISTERED_BATCH_SIZE,
                learning_rate=bridge._registered_learning_rate(0),
                batch_mean_training_loss=np.float32(0.5),
                returned_preclip_l2_norm=np.float32(1.0),
                optimizer_generation_before=0,
                optimizer_generation_after=1,
                model_sha256_before="0" * 64,
                model_sha256_after="1" * 64,
                optimizer_sha256_before="2" * 64,
                optimizer_sha256_after="3" * 64,
                rng_sha256_before="4" * 64,
                rng_sha256_after="5" * 64,
            )
            consumption = bridge._mark_transition_trace_consumed(
                transition,
                receipt_token=object(),
                trace_session_token=object(),
                trace_record_index=0,
            )
            return transition, consumption

        local = bridge._issue_registered_executor_session(20_260_720)
        local_transition, local_consumption = trace_consumed(local)
        foreign = bridge._issue_registered_executor_session(20_260_721)
        foreign_transition, foreign_consumption = trace_consumed(foreign)

        local_state = bridge._ISSUED_TRANSITIONS[local_transition]
        local_state.trace_consumption = foreign_consumption
        bridge._TRANSITION_CONSUMPTIONS[local_transition] = foreign_consumption
        try:
            bridge._transition_snapshot(local_transition)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("two foreign consumption links won consensus")
        assert bridge._EXECUTOR_SESSIONS[local].failed
        assert local_state.invalid
        local_consumption_state = bridge._ISSUED_CONSUMPTIONS[local_consumption]
        assert local_consumption_state.invalid
        assert local_consumption_state.failed
        foreign_consumption_state = bridge._ISSUED_CONSUMPTIONS[foreign_consumption]
        assert not foreign_consumption_state.invalid
        assert not foreign_consumption_state.failed
        assert not bridge._EXECUTOR_SESSIONS[foreign].failed
        assert bridge._ISSUED_TRANSITIONS[foreign_transition].phase == "TRACE_CONSUMED"
        bridge._trace_consumption_snapshot(foreign_consumption, required_used=False)
        bridge._executor_session_snapshot(foreign)
        """
    )


@pytest.mark.parametrize(
    "failing_registry",
    (
        "transition_consumptions",
        "transition_guards",
        "consumption_guards",
        "session_guards",
    ),
)
def test_persistent_registry_read_failure_still_poison_closes_owner(
    failing_registry: str,
) -> None:
    _run_isolated_bridge_probe(
        f"""
        import numpy as np
        import falsewake.experiment_002_training_bridge as bridge

        class FailingGet:
            def __init__(self, backing):
                self.backing = backing
            def get(self, key, default=None):
                raise RuntimeError("persistent registry read failure")
            def __getitem__(self, key):
                return self.backing[key]
            def __setitem__(self, key, value):
                self.backing[key] = value
            def __contains__(self, key):
                return key in self.backing
            def __iter__(self):
                return iter(self.backing)
            def __len__(self):
                return len(self.backing)
            def items(self):
                return self.backing.items()
            def values(self):
                return self.backing.values()
            def pop(self, key, default=None):
                return self.backing.pop(key, default)

        authority = bridge._issue_registered_executor_session(20_260_720)
        epoch_token = object()
        batch_token = object()

        def issue(update):
            return bridge._issue_registered_optimizer_transition(
                executor_authority=authority,
                epoch_session_token=epoch_token,
                batch_token=batch_token if update == 0 else object(),
                zero_based_epoch=0,
                zero_based_global_update=update,
                batch_size=bridge._REGISTERED_BATCH_SIZE,
                learning_rate=bridge._registered_learning_rate(update),
                batch_mean_training_loss=np.float32(0.5),
                returned_preclip_l2_norm=np.float32(1.0),
                optimizer_generation_before=update,
                optimizer_generation_after=update + 1,
                model_sha256_before="0" * 64 if update == 0 else "1" * 64,
                model_sha256_after="1" * 64 if update == 0 else "6" * 64,
                optimizer_sha256_before="2" * 64 if update == 0 else "3" * 64,
                optimizer_sha256_after="3" * 64 if update == 0 else "7" * 64,
                rng_sha256_before="4" * 64 if update == 0 else "5" * 64,
                rng_sha256_after="5" * 64 if update == 0 else "8" * 64,
            )

        transition = issue(0)
        receipt_token = object()
        consumption = bridge._mark_transition_trace_consumed(
            transition,
            receipt_token=receipt_token,
            trace_session_token=object(),
            trace_record_index=0,
        )
        failing_registry = {failing_registry!r}
        if failing_registry != "session_guards":
            bridge._accept_trace_consumed_transition(
                transition,
                consumption,
                receipt_token=receipt_token,
                executor_authority=authority,
                epoch_session_token=epoch_token,
                batch_token=batch_token,
            )

        if failing_registry == "transition_consumptions":
            original = bridge._TRANSITION_CONSUMPTIONS
            bridge._TRANSITION_CONSUMPTIONS = FailingGet(original)
            accessor = lambda: bridge._transition_snapshot(transition)
        elif failing_registry == "transition_guards":
            original = bridge._TRANSITION_GUARDS
            bridge._TRANSITION_GUARDS = FailingGet(original)
            accessor = lambda: bridge._transition_snapshot(transition)
        elif failing_registry == "consumption_guards":
            original = bridge._CONSUMPTION_GUARDS
            bridge._CONSUMPTION_GUARDS = FailingGet(original)
            accessor = lambda: bridge._trace_consumption_snapshot(consumption)
        elif failing_registry == "session_guards":
            original = bridge._EXECUTOR_SESSION_GUARDS
            bridge._EXECUTOR_SESSION_GUARDS = FailingGet(original)
            accessor = lambda: bridge._executor_session_snapshot(authority)
        else:
            raise AssertionError("unknown registry")
        try:
            accessor()
        except BaseException as error:
            assert "registry read" in str(error)
        else:
            raise AssertionError("persistent read failure was hidden")
        if failing_registry == "transition_consumptions":
            bridge._TRANSITION_CONSUMPTIONS = original
        elif failing_registry == "transition_guards":
            bridge._TRANSITION_GUARDS = original
        elif failing_registry == "consumption_guards":
            bridge._CONSUMPTION_GUARDS = original
        elif failing_registry == "session_guards":
            bridge._EXECUTOR_SESSION_GUARDS = original

        assert bridge._EXECUTOR_SESSIONS[authority].failed
        assert bridge._ISSUED_TRANSITIONS[transition].invalid
        assert bridge._ISSUED_CONSUMPTIONS[consumption].invalid
        try:
            bridge._trace_consumption_snapshot(consumption)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("consumption retried after persistent read failure")
        try:
            issue(1)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("owner continued after persistent read failure")
        """
    )


@pytest.mark.parametrize(
    "primary_registry",
    ("executor_sessions", "issued_transitions", "issued_consumptions"),
)
@pytest.mark.parametrize("failure_mode", ("persistent_get", "missing_row"))
def test_primary_registry_read_failure_poison_closes_capability(
    primary_registry: str,
    failure_mode: str,
) -> None:
    _run_isolated_bridge_probe(
        f"""
        import numpy as np
        import falsewake.experiment_002_training_bridge as bridge

        class FailingGet:
            def __init__(self, backing):
                self.backing = backing
            def get(self, key, default=None):
                raise RuntimeError("persistent primary registry read failure")
            def __getitem__(self, key):
                return self.backing[key]
            def __setitem__(self, key, value):
                self.backing[key] = value
            def __contains__(self, key):
                return key in self.backing
            def __iter__(self):
                return iter(self.backing)
            def __len__(self):
                return len(self.backing)
            def items(self):
                return self.backing.items()
            def values(self):
                return self.backing.values()
            def pop(self, key, default=None):
                return self.backing.pop(key, default)

        authority = bridge._issue_registered_executor_session(20_260_720)
        epoch_token = object()
        batch_token = object()

        def issue(update):
            return bridge._issue_registered_optimizer_transition(
                executor_authority=authority,
                epoch_session_token=epoch_token,
                batch_token=batch_token if update == 0 else object(),
                zero_based_epoch=0,
                zero_based_global_update=update,
                batch_size=bridge._REGISTERED_BATCH_SIZE,
                learning_rate=bridge._registered_learning_rate(update),
                batch_mean_training_loss=np.float32(0.5),
                returned_preclip_l2_norm=np.float32(1.0),
                optimizer_generation_before=update,
                optimizer_generation_after=update + 1,
                model_sha256_before="0" * 64 if update == 0 else "1" * 64,
                model_sha256_after="1" * 64 if update == 0 else "6" * 64,
                optimizer_sha256_before="2" * 64 if update == 0 else "3" * 64,
                optimizer_sha256_after="3" * 64 if update == 0 else "7" * 64,
                rng_sha256_before="4" * 64 if update == 0 else "5" * 64,
                rng_sha256_after="5" * 64 if update == 0 else "8" * 64,
            )

        transition = issue(0)
        receipt_token = object()
        consumption = bridge._mark_transition_trace_consumed(
            transition,
            receipt_token=receipt_token,
            trace_session_token=object(),
            trace_record_index=0,
        )
        primary_registry = {primary_registry!r}
        failure_mode = {failure_mode!r}
        if primary_registry != "executor_sessions":
            bridge._accept_trace_consumed_transition(
                transition,
                consumption,
                receipt_token=receipt_token,
                executor_authority=authority,
                epoch_session_token=epoch_token,
                batch_token=batch_token,
            )

        if primary_registry == "executor_sessions":
            original = bridge._EXECUTOR_SESSIONS
            primary_key = authority
            accessor = lambda: bridge._executor_session_snapshot(authority)
        elif primary_registry == "issued_transitions":
            original = bridge._ISSUED_TRANSITIONS
            primary_key = transition
            accessor = lambda: bridge._transition_snapshot(transition)
        elif primary_registry == "issued_consumptions":
            original = bridge._ISSUED_CONSUMPTIONS
            primary_key = consumption
            accessor = lambda: bridge._trace_consumption_snapshot(consumption)
        else:
            raise AssertionError("unknown primary registry")
        if failure_mode == "persistent_get":
            failing = FailingGet(original)
            if primary_registry == "executor_sessions":
                bridge._EXECUTOR_SESSIONS = failing
            elif primary_registry == "issued_transitions":
                bridge._ISSUED_TRANSITIONS = failing
            else:
                bridge._ISSUED_CONSUMPTIONS = failing
        elif failure_mode == "missing_row":
            saved_state = original.pop(primary_key)
        else:
            raise AssertionError("unknown primary failure mode")
        try:
            accessor()
        except RuntimeError as error:
            assert failure_mode == "persistent_get"
            assert "primary registry read" in str(error)
        except bridge.Experiment002TrainingBridgeError:
            assert failure_mode == "missing_row"
        else:
            raise AssertionError("primary read failure was hidden")
        finally:
            if failure_mode == "missing_row":
                original[primary_key] = saved_state
            elif primary_registry == "executor_sessions":
                bridge._EXECUTOR_SESSIONS = original
            elif primary_registry == "issued_transitions":
                bridge._ISSUED_TRANSITIONS = original
            else:
                bridge._ISSUED_CONSUMPTIONS = original

        for accessor in (
            lambda: bridge._executor_session_snapshot(authority),
            lambda: bridge._transition_snapshot(transition),
            lambda: bridge._trace_consumption_snapshot(consumption),
        ):
            try:
                accessor()
            except bridge.Experiment002TrainingBridgeError:
                pass
            else:
                raise AssertionError("capability resurrected after registry restore")
        session_state = bridge._EXECUTOR_SESSIONS[authority]
        transition_state = bridge._ISSUED_TRANSITIONS[transition]
        consumption_state = bridge._ISSUED_CONSUMPTIONS[consumption]
        assert session_state.failed
        assert session_state.active_transition_token is None
        assert transition_state.invalid
        assert consumption_state.invalid
        if primary_registry == "executor_sessions":
            assert transition_state.phase == "FAILED"
            assert consumption_state.failed and not consumption_state.used
        else:
            assert transition_state.phase == "POPULATION_ACCEPTED"
            assert consumption_state.used and not consumption_state.failed
        try:
            issue(1)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("owner continued after primary read failure")
        """
    )


@pytest.mark.parametrize(
    "primary_registry",
    ("executor_sessions", "issued_transitions", "issued_consumptions"),
)
@pytest.mark.parametrize("access_path", ("direct", "nested"))
def test_foreign_primary_row_never_cross_poisons_its_owner(
    primary_registry: str,
    access_path: str,
) -> None:
    _run_isolated_bridge_probe(
        f"""
        import numpy as np
        import falsewake.experiment_002_training_bridge as bridge

        def issue(authority, epoch_token, batch_token):
            return bridge._issue_registered_optimizer_transition(
                executor_authority=authority,
                epoch_session_token=epoch_token,
                batch_token=batch_token,
                zero_based_epoch=0,
                zero_based_global_update=0,
                batch_size=bridge._REGISTERED_BATCH_SIZE,
                learning_rate=bridge._registered_learning_rate(0),
                batch_mean_training_loss=np.float32(0.5),
                returned_preclip_l2_norm=np.float32(1.0),
                optimizer_generation_before=0,
                optimizer_generation_after=1,
                model_sha256_before="0" * 64,
                model_sha256_after="1" * 64,
                optimizer_sha256_before="2" * 64,
                optimizer_sha256_after="3" * 64,
                rng_sha256_before="4" * 64,
                rng_sha256_after="5" * 64,
            )

        local = bridge._issue_registered_executor_session(20_260_720)
        local_epoch = object()
        local_batch = object()
        local_transition = issue(local, local_epoch, local_batch)
        local_receipt = object()
        local_consumption = bridge._mark_transition_trace_consumed(
            local_transition,
            receipt_token=local_receipt,
            trace_session_token=object(),
            trace_record_index=0,
        )

        foreign = bridge._issue_registered_executor_session(20_260_721)
        foreign_epoch = object()
        foreign_batch = object()
        foreign_transition = issue(foreign, foreign_epoch, foreign_batch)
        foreign_receipt = object()
        foreign_consumption = bridge._mark_transition_trace_consumed(
            foreign_transition,
            receipt_token=foreign_receipt,
            trace_session_token=object(),
            trace_record_index=0,
        )

        primary_registry = {primary_registry!r}
        access_path = {access_path!r}
        if primary_registry == "executor_sessions":
            registry = bridge._EXECUTOR_SESSIONS
            local_key = local
            foreign_key = foreign
            direct_accessor = lambda: bridge._executor_session_snapshot(local)
            nested_accessor = lambda: bridge._transition_snapshot(local_transition)
        elif primary_registry == "issued_transitions":
            registry = bridge._ISSUED_TRANSITIONS
            local_key = local_transition
            foreign_key = foreign_transition
            direct_accessor = lambda: bridge._transition_snapshot(local_transition)
            nested_accessor = lambda: bridge._executor_session_snapshot(local)
        elif primary_registry == "issued_consumptions":
            registry = bridge._ISSUED_CONSUMPTIONS
            local_key = local_consumption
            foreign_key = foreign_consumption
            direct_accessor = lambda: bridge._trace_consumption_snapshot(
                local_consumption
            )
            nested_accessor = lambda: bridge._transition_snapshot(local_transition)
        else:
            raise AssertionError("unknown primary registry")
        saved_local_state = registry[local_key]
        registry[local_key] = registry[foreign_key]
        try:
            accessor = direct_accessor if access_path == "direct" else nested_accessor
            accessor()
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("foreign primary row was accepted")
        finally:
            registry[local_key] = saved_local_state

        for accessor in (
            lambda: bridge._executor_session_snapshot(local),
            lambda: bridge._transition_snapshot(local_transition),
            lambda: bridge._trace_consumption_snapshot(local_consumption),
        ):
            try:
                accessor()
            except bridge.Experiment002TrainingBridgeError:
                pass
            else:
                raise AssertionError("local capability survived foreign-row corruption")
        assert bridge._EXECUTOR_SESSIONS[local].failed
        assert bridge._ISSUED_TRANSITIONS[local_transition].invalid
        assert bridge._ISSUED_CONSUMPTIONS[local_consumption].invalid

        foreign_session_state = bridge._EXECUTOR_SESSIONS[foreign]
        foreign_transition_state = bridge._ISSUED_TRANSITIONS[foreign_transition]
        foreign_consumption_state = bridge._ISSUED_CONSUMPTIONS[foreign_consumption]
        assert not foreign_session_state.failed
        assert foreign_transition_state.phase == "TRACE_CONSUMED"
        assert not foreign_transition_state.invalid
        assert not foreign_consumption_state.used
        assert not foreign_consumption_state.failed
        assert not foreign_consumption_state.invalid
        bridge._executor_session_snapshot(foreign)
        bridge._transition_snapshot(foreign_transition, required_phase="TRACE_CONSUMED")
        bridge._trace_consumption_snapshot(foreign_consumption, required_used=False)
        bridge._accept_trace_consumed_transition(
            foreign_transition,
            foreign_consumption,
            receipt_token=foreign_receipt,
            executor_authority=foreign,
            epoch_session_token=foreign_epoch,
            batch_token=foreign_batch,
        )
        assert bridge._ISSUED_TRANSITIONS[foreign_transition].phase == (
            "POPULATION_ACCEPTED"
        )
        assert bridge._ISSUED_CONSUMPTIONS[foreign_consumption].used
        assert not bridge._EXECUTOR_SESSIONS[foreign].failed
        """
    )


@pytest.mark.parametrize(
    "coherent_alias",
    ("registered_transition", "consumption", "same_owner_transition"),
)
def test_two_link_foreign_alias_is_rejected_without_cross_poison(
    coherent_alias: str,
) -> None:
    _run_isolated_bridge_probe(
        f"""
        import numpy as np
        import falsewake.experiment_002_training_bridge as bridge

        def issue(authority, epoch_token, batch_token, update):
            return bridge._issue_registered_optimizer_transition(
                executor_authority=authority,
                epoch_session_token=epoch_token,
                batch_token=batch_token,
                zero_based_epoch=0,
                zero_based_global_update=update,
                batch_size=bridge._REGISTERED_BATCH_SIZE,
                learning_rate=bridge._registered_learning_rate(update),
                batch_mean_training_loss=np.float32(0.5),
                returned_preclip_l2_norm=np.float32(1.0),
                optimizer_generation_before=update,
                optimizer_generation_after=update + 1,
                model_sha256_before="0" * 64 if update == 0 else "1" * 64,
                model_sha256_after="1" * 64 if update == 0 else "6" * 64,
                optimizer_sha256_before="2" * 64 if update == 0 else "3" * 64,
                optimizer_sha256_after="3" * 64 if update == 0 else "7" * 64,
                rng_sha256_before="4" * 64 if update == 0 else "5" * 64,
                rng_sha256_after="5" * 64 if update == 0 else "8" * 64,
            )

        def consume(transition, receipt, index):
            return bridge._mark_transition_trace_consumed(
                transition,
                receipt_token=receipt,
                trace_session_token=object(),
                trace_record_index=index,
            )

        local = bridge._issue_registered_executor_session(20_260_720)
        local_epoch = object()
        local_batch = object()
        local_transition = issue(local, local_epoch, local_batch, 0)
        local_receipt = object()
        local_consumption = consume(local_transition, local_receipt, 0)

        foreign = bridge._issue_registered_executor_session(20_260_721)
        foreign_epoch = object()
        foreign_batch = object()
        foreign_transition = issue(foreign, foreign_epoch, foreign_batch, 0)
        foreign_receipt = object()
        foreign_consumption = consume(foreign_transition, foreign_receipt, 0)

        coherent_alias = {coherent_alias!r}
        if coherent_alias == "registered_transition":
            primary = bridge._ISSUED_TRANSITIONS
            guards = bridge._TRANSITION_GUARDS
            local_key = local_transition
            foreign_key = foreign_transition
            saved_primary = primary[local_key]
            saved_guard = guards[local_key]
            primary[local_key] = primary[foreign_key]
            guards[local_key] = guards[foreign_key]
            accessor = lambda: bridge._transition_snapshot(local_transition)
        elif coherent_alias == "consumption":
            primary = bridge._ISSUED_CONSUMPTIONS
            guards = bridge._CONSUMPTION_GUARDS
            local_key = local_consumption
            foreign_key = foreign_consumption
            saved_primary = primary[local_key]
            saved_guard = guards[local_key]
            primary[local_key] = primary[foreign_key]
            guards[local_key] = guards[foreign_key]
            accessor = lambda: bridge._trace_consumption_snapshot(local_consumption)
        elif coherent_alias == "same_owner_transition":
            bridge._accept_trace_consumed_transition(
                local_transition,
                local_consumption,
                receipt_token=local_receipt,
                executor_authority=local,
                epoch_session_token=local_epoch,
                batch_token=local_batch,
            )
            next_transition = issue(local, local_epoch, object(), 1)
            primary = bridge._ISSUED_TRANSITIONS
            guards = bridge._TRANSITION_GUARDS
            local_key = next_transition
            foreign_key = local_transition
            saved_primary = primary[local_key]
            saved_guard = guards[local_key]
            primary[local_key] = primary[foreign_key]
            guards[local_key] = guards[foreign_key]
            accessor = lambda: bridge._transition_snapshot(next_transition)
        else:
            raise AssertionError("unknown coherent alias")
        try:
            accessor()
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("coherent two-link alias was accepted")
        finally:
            primary[local_key] = saved_primary
            guards[local_key] = saved_guard

        if coherent_alias == "same_owner_transition":
            try:
                bridge._transition_snapshot(next_transition)
            except bridge.Experiment002TrainingBridgeError:
                pass
            else:
                raise AssertionError("same-owner aliased transition resurrected")
            bridge._transition_snapshot(
                local_transition, required_phase="POPULATION_ACCEPTED"
            )
            bridge._trace_consumption_snapshot(local_consumption, required_used=True)
            assert not bridge._ISSUED_TRANSITIONS[local_transition].invalid
            assert not bridge._ISSUED_CONSUMPTIONS[local_consumption].invalid
            assert bridge._ISSUED_TRANSITIONS[next_transition].invalid
            try:
                issue(local, local_epoch, object(), 2)
            except bridge.Experiment002TrainingBridgeError:
                pass
            else:
                raise AssertionError("terminal same-owner session issued t2")
        else:
            for local_accessor in (
                lambda: bridge._executor_session_snapshot(local),
                lambda: bridge._transition_snapshot(local_transition),
                lambda: bridge._trace_consumption_snapshot(local_consumption),
            ):
                try:
                    local_accessor()
                except bridge.Experiment002TrainingBridgeError:
                    pass
                else:
                    raise AssertionError("local aliased capability resurrected")
        assert bridge._EXECUTOR_SESSIONS[local].failed

        assert not bridge._EXECUTOR_SESSIONS[foreign].failed
        assert not bridge._ISSUED_TRANSITIONS[foreign_transition].invalid
        assert not bridge._ISSUED_CONSUMPTIONS[foreign_consumption].invalid
        bridge._executor_session_snapshot(foreign)
        bridge._transition_snapshot(foreign_transition, required_phase="TRACE_CONSUMED")
        bridge._trace_consumption_snapshot(foreign_consumption, required_used=False)
        """
    )


def test_two_link_synthetic_alias_is_rejected_without_cross_poison() -> None:
    _run_isolated_bridge_probe(
        """
        import numpy as np
        import falsewake.experiment_002_training_bridge as bridge

        def issue(seed):
            return bridge._issue_synthetic_optimizer_transition(
                seed=seed,
                zero_based_epoch=0,
                zero_based_global_update=0,
                batch_size=2,
                learning_rate=0.1,
                batch_mean_training_loss=np.float32(0.5),
                returned_preclip_l2_norm=np.float32(1.0),
                optimizer_generation_before=0,
                optimizer_generation_after=1,
                model_sha256_before="0" * 64,
                model_sha256_after="1" * 64,
                optimizer_sha256_before="2" * 64,
                optimizer_sha256_after="3" * 64,
                rng_sha256_before="4" * 64,
                rng_sha256_after="5" * 64,
            )

        local = issue(7)
        foreign = issue(8)
        local_state = bridge._ISSUED_TRANSITIONS[local]
        local_guard = bridge._TRANSITION_GUARDS[local]
        foreign_state = bridge._ISSUED_TRANSITIONS[foreign]
        foreign_guard = bridge._TRANSITION_GUARDS[foreign]
        bridge._ISSUED_TRANSITIONS[local] = foreign_state
        bridge._TRANSITION_GUARDS[local] = foreign_guard
        try:
            bridge._transition_snapshot(local)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("synthetic two-link alias was accepted")
        finally:
            bridge._ISSUED_TRANSITIONS[local] = local_state
            bridge._TRANSITION_GUARDS[local] = local_guard

        try:
            bridge._transition_snapshot(local)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("synthetic local capability resurrected")
        assert local in bridge._INVALID_TRANSITIONS
        assert not foreign_state.invalid
        assert foreign not in bridge._INVALID_TRANSITIONS
        bridge._transition_snapshot(foreign, required_phase="ISSUED")
        """
    )


def test_nested_primary_transition_read_failure_poison_closes_accepted_owner() -> None:
    _run_isolated_bridge_probe(
        """
        import numpy as np
        import falsewake.experiment_002_training_bridge as bridge

        class FailingGet:
            def __init__(self, backing):
                self.backing = backing
            def get(self, key, default=None):
                raise RuntimeError("nested primary transition read failure")
            def __getitem__(self, key):
                return self.backing[key]
            def __setitem__(self, key, value):
                self.backing[key] = value
            def __contains__(self, key):
                return key in self.backing
            def __iter__(self):
                return iter(self.backing)
            def __len__(self):
                return len(self.backing)
            def items(self):
                return self.backing.items()
            def values(self):
                return self.backing.values()
            def pop(self, key, default=None):
                return self.backing.pop(key, default)

        authority = bridge._issue_registered_executor_session(20_260_720)
        epoch_token = object()
        batch_token = object()

        def issue(update):
            return bridge._issue_registered_optimizer_transition(
                executor_authority=authority,
                epoch_session_token=epoch_token,
                batch_token=batch_token if update == 0 else object(),
                zero_based_epoch=0,
                zero_based_global_update=update,
                batch_size=bridge._REGISTERED_BATCH_SIZE,
                learning_rate=bridge._registered_learning_rate(update),
                batch_mean_training_loss=np.float32(0.5),
                returned_preclip_l2_norm=np.float32(1.0),
                optimizer_generation_before=update,
                optimizer_generation_after=update + 1,
                model_sha256_before="0" * 64 if update == 0 else "1" * 64,
                model_sha256_after="1" * 64 if update == 0 else "6" * 64,
                optimizer_sha256_before="2" * 64 if update == 0 else "3" * 64,
                optimizer_sha256_after="3" * 64 if update == 0 else "7" * 64,
                rng_sha256_before="4" * 64 if update == 0 else "5" * 64,
                rng_sha256_after="5" * 64 if update == 0 else "8" * 64,
            )

        transition = issue(0)
        receipt_token = object()
        consumption = bridge._mark_transition_trace_consumed(
            transition,
            receipt_token=receipt_token,
            trace_session_token=object(),
            trace_record_index=0,
        )
        bridge._accept_trace_consumed_transition(
            transition,
            consumption,
            receipt_token=receipt_token,
            executor_authority=authority,
            epoch_session_token=epoch_token,
            batch_token=batch_token,
        )

        original = bridge._ISSUED_TRANSITIONS
        bridge._ISSUED_TRANSITIONS = FailingGet(original)
        try:
            bridge._trace_consumption_snapshot(consumption, required_used=True)
        except RuntimeError as error:
            assert "nested primary transition read" in str(error)
        else:
            raise AssertionError("nested primary read failure was hidden")
        finally:
            bridge._ISSUED_TRANSITIONS = original

        session_state = bridge._EXECUTOR_SESSIONS[authority]
        transition_state = bridge._ISSUED_TRANSITIONS[transition]
        consumption_state = bridge._ISSUED_CONSUMPTIONS[consumption]
        assert session_state.failed
        assert session_state.active_transition_token is None
        assert transition_state.invalid
        assert transition_state.phase == "POPULATION_ACCEPTED"
        assert consumption_state.invalid
        assert consumption_state.used and not consumption_state.failed
        try:
            bridge._trace_consumption_snapshot(consumption, required_used=True)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("consumption resurrected after registry restore")
        try:
            issue(1)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("accepted owner continued after nested read failure")
        """
    )


def test_missing_session_guard_still_terminalizes_owned_active_transition() -> None:
    _run_isolated_bridge_probe(
        """
        import numpy as np
        import falsewake.experiment_002_training_bridge as bridge

        authority = bridge._issue_registered_executor_session(20_260_720)
        transition = bridge._issue_registered_optimizer_transition(
            executor_authority=authority,
            epoch_session_token=object(),
            batch_token=object(),
            zero_based_epoch=0,
            zero_based_global_update=0,
            batch_size=bridge._REGISTERED_BATCH_SIZE,
            learning_rate=bridge._registered_learning_rate(0),
            batch_mean_training_loss=np.float32(0.5),
            returned_preclip_l2_norm=np.float32(1.0),
            optimizer_generation_before=0,
            optimizer_generation_after=1,
            model_sha256_before="0" * 64,
            model_sha256_after="1" * 64,
            optimizer_sha256_before="2" * 64,
            optimizer_sha256_after="3" * 64,
            rng_sha256_before="4" * 64,
            rng_sha256_after="5" * 64,
        )
        consumption = bridge._mark_transition_trace_consumed(
            transition,
            receipt_token=object(),
            trace_session_token=object(),
            trace_record_index=0,
        )
        saved_guard = bridge._EXECUTOR_SESSION_GUARDS.pop(authority)
        try:
            bridge._executor_session_snapshot(authority)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("missing session guard was ignored")
        bridge._EXECUTOR_SESSION_GUARDS[authority] = saved_guard
        assert bridge._EXECUTOR_SESSIONS[authority].failed
        assert bridge._ISSUED_TRANSITIONS[transition].phase == "FAILED"
        consumption_state = bridge._ISSUED_CONSUMPTIONS[consumption]
        assert consumption_state.failed and not consumption_state.used
        try:
            bridge._trace_consumption_snapshot(consumption)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("owned consumption retried after session failure")
        """
    )


@pytest.mark.parametrize(
    "session_corruption",
    (
        "active_none",
        "active_foreign",
        "transition_owner_foreign",
        "transition_token_foreign",
    ),
)
def test_session_cleanup_uses_anchor_generation_not_corrupt_raw_links(
    session_corruption: str,
) -> None:
    _run_isolated_bridge_probe(
        f"""
        from dataclasses import replace
        import numpy as np
        import falsewake.experiment_002_training_bridge as bridge

        def issue(authority):
            return bridge._issue_registered_optimizer_transition(
                executor_authority=authority,
                epoch_session_token=object(),
                batch_token=object(),
                zero_based_epoch=0,
                zero_based_global_update=0,
                batch_size=bridge._REGISTERED_BATCH_SIZE,
                learning_rate=bridge._registered_learning_rate(0),
                batch_mean_training_loss=np.float32(0.5),
                returned_preclip_l2_norm=np.float32(1.0),
                optimizer_generation_before=0,
                optimizer_generation_after=1,
                model_sha256_before="0" * 64,
                model_sha256_after="1" * 64,
                optimizer_sha256_before="2" * 64,
                optimizer_sha256_after="3" * 64,
                rng_sha256_before="4" * 64,
                rng_sha256_after="5" * 64,
            )

        local = bridge._issue_registered_executor_session(20_260_720)
        local_transition = issue(local)
        local_consumption = bridge._mark_transition_trace_consumed(
            local_transition,
            receipt_token=object(),
            trace_session_token=object(),
            trace_record_index=0,
        )
        foreign = bridge._issue_registered_executor_session(20_260_721)
        foreign_transition = issue(foreign)
        foreign_token = (
            bridge._ISSUED_TRANSITIONS[foreign_transition].issuer.transition_token
        )
        local_state = bridge._EXECUTOR_SESSIONS[local]
        corruption = {session_corruption!r}
        if corruption == "active_none":
            local_state.active_transition_token = None
        elif corruption == "active_foreign":
            local_state.active_transition_token = foreign_token
            bridge._EXECUTOR_SESSION_LIFECYCLES[local] = (
                bridge._ExecutorSessionLifecycleGuard(1, foreign_token, False)
            )
        elif corruption == "transition_owner_foreign":
            transition_state = bridge._ISSUED_TRANSITIONS[local_transition]
            transition_state.issuer = replace(
                transition_state.issuer,
                executor_authority=foreign,
            )
        elif corruption == "transition_token_foreign":
            transition_state = bridge._ISSUED_TRANSITIONS[local_transition]
            transition_state.issuer = replace(
                transition_state.issuer,
                transition_token=foreign_token,
            )
        else:
            raise AssertionError("unknown session corruption")
        try:
            bridge._executor_session_snapshot(local)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("corrupt session links were ignored")
        assert local_state.failed
        assert bridge._ISSUED_TRANSITIONS[local_transition].phase == "FAILED"
        assert bridge._ISSUED_TRANSITIONS[local_transition].invalid
        local_consumption_state = bridge._ISSUED_CONSUMPTIONS[local_consumption]
        assert local_consumption_state.failed
        assert local_consumption_state.invalid
        assert bridge._ISSUED_TRANSITIONS[foreign_transition].phase == "ISSUED"
        assert not bridge._EXECUTOR_SESSIONS[foreign].failed
        bridge._transition_snapshot(foreign_transition, required_phase="ISSUED")
        bridge._executor_session_snapshot(foreign)
        """
    )


def test_session_terminalization_never_fails_foreign_active_token() -> None:
    _run_isolated_bridge_probe(
        """
        import numpy as np
        import falsewake.experiment_002_training_bridge as bridge

        first = bridge._issue_registered_executor_session(20_260_720)
        second = bridge._issue_registered_executor_session(20_260_721)
        transition = bridge._issue_registered_optimizer_transition(
            executor_authority=second,
            epoch_session_token=object(),
            batch_token=object(),
            zero_based_epoch=0,
            zero_based_global_update=0,
            batch_size=bridge._REGISTERED_BATCH_SIZE,
            learning_rate=bridge._registered_learning_rate(0),
            batch_mean_training_loss=np.float32(0.5),
            returned_preclip_l2_norm=np.float32(1.0),
            optimizer_generation_before=0,
            optimizer_generation_after=1,
            model_sha256_before="0" * 64,
            model_sha256_after="1" * 64,
            optimizer_sha256_before="2" * 64,
            optimizer_sha256_after="3" * 64,
            rng_sha256_before="4" * 64,
            rng_sha256_after="5" * 64,
        )
        foreign_token = bridge._ISSUED_TRANSITIONS[transition].issuer.transition_token
        first_state = bridge._EXECUTOR_SESSIONS[first]
        first_state.active_transition_token = foreign_token
        bridge._EXECUTOR_SESSION_LIFECYCLES[first] = (
            bridge._ExecutorSessionLifecycleGuard(0, foreign_token, False)
        )
        try:
            bridge._executor_session_snapshot(first)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("foreign active token corruption was ignored")
        assert first_state.failed
        assert bridge._ISSUED_TRANSITIONS[transition].phase == "ISSUED"
        assert not bridge._EXECUTOR_SESSIONS[second].failed
        bridge._transition_snapshot(transition, required_phase="ISSUED")
        bridge._executor_session_snapshot(second)
        """
    )


def test_registered_bridge_is_one_use_atomic_and_cross_session_safe() -> None:
    first = bridge._issue_registered_executor_session(20_260_720)
    second = bridge._issue_registered_executor_session(20_260_721)
    with pytest.raises(bridge.Experiment002TrainingBridgeError, match="already"):
        bridge._issue_registered_executor_session(20_260_720)

    epoch_a = object()
    batch_a0 = object()
    receipt_a0 = object()
    transition_a0 = _registered_transition(
        first,
        epoch_token=epoch_a,
        batch_token=batch_a0,
        global_update=0,
    )
    consumption_a0 = _accept(
        first,
        transition_a0,
        epoch_token=epoch_a,
        batch_token=batch_a0,
        receipt_token=receipt_a0,
    )
    assert transition_a0.phase == "POPULATION_ACCEPTED"
    bridge._trace_consumption_snapshot(consumption_a0, required_used=True)
    with pytest.raises(bridge.Experiment002TrainingBridgeError):
        bridge._mark_transition_trace_consumed(
            transition_a0,
            receipt_token=receipt_a0,
            trace_session_token=object(),
            trace_record_index=0,
        )
    assert transition_a0.phase == "POPULATION_ACCEPTED"

    batch_a1 = object()
    transition_a1 = _registered_transition(
        first,
        epoch_token=epoch_a,
        batch_token=batch_a1,
        global_update=1,
    )
    receipt_a1 = object()
    consumption_a1 = bridge._mark_transition_trace_consumed(
        transition_a1,
        receipt_token=receipt_a1,
        trace_session_token=object(),
        trace_record_index=1,
    )
    epoch_b = object()
    batch_b0 = object()
    transition_b0 = _registered_transition(
        second,
        epoch_token=epoch_b,
        batch_token=batch_b0,
        global_update=0,
    )
    receipt_b0 = object()
    consumption_b0 = bridge._mark_transition_trace_consumed(
        transition_b0,
        receipt_token=receipt_b0,
        trace_session_token=object(),
        trace_record_index=0,
    )
    with pytest.raises(bridge.Experiment002TrainingBridgeError, match="claimed"):
        bridge._accept_trace_consumed_transition(
            transition_a1,
            consumption_b0,
            receipt_token=receipt_a1,
            executor_authority=first,
            epoch_session_token=epoch_a,
            batch_token=batch_a1,
        )
    assert transition_a1.phase == "TRACE_CONSUMED"
    assert transition_b0.phase == "TRACE_CONSUMED"
    bridge._trace_consumption_snapshot(consumption_a1, required_used=False)
    bridge._trace_consumption_snapshot(consumption_b0, required_used=False)
    bridge._accept_trace_consumed_transition(
        transition_a1,
        consumption_a1,
        receipt_token=receipt_a1,
        executor_authority=first,
        epoch_session_token=epoch_a,
        batch_token=batch_a1,
    )
    bridge._accept_trace_consumed_transition(
        transition_b0,
        consumption_b0,
        receipt_token=receipt_b0,
        executor_authority=second,
        epoch_session_token=epoch_b,
        batch_token=batch_b0,
    )

    accepted_state = bridge._ISSUED_TRANSITIONS[transition_a0]
    accepted_state.phase = "ISSUED"
    accepted_state.trace_authority = None
    accepted_state.trace_consumption = None
    bridge._TRANSITION_PHASES[transition_a0] = "ISSUED"
    bridge._TRANSITION_CONSUMPTIONS.pop(transition_a0, None)
    with pytest.raises(bridge.Experiment002TrainingBridgeError, match="issuer state"):
        bridge._transition_snapshot(transition_a0)
    with pytest.raises(bridge.Experiment002TrainingBridgeError):
        bridge._trace_consumption_snapshot(consumption_a0, required_used=True)


def test_persistent_mark_phase_registry_failure_is_terminal() -> None:
    _run_isolated_bridge_probe(
        """
        import weakref
        import numpy as np
        import falsewake.experiment_002_training_bridge as bridge

        authority = bridge._issue_registered_executor_session(20_260_720)
        epoch_token = object()
        batch_token = object()
        transition = bridge._issue_registered_optimizer_transition(
            executor_authority=authority,
            epoch_session_token=epoch_token,
            batch_token=batch_token,
            zero_based_epoch=0,
            zero_based_global_update=0,
            batch_size=bridge._REGISTERED_BATCH_SIZE,
            learning_rate=bridge._registered_learning_rate(0),
            batch_mean_training_loss=np.float32(0.5),
            returned_preclip_l2_norm=np.float32(1.0),
            optimizer_generation_before=0,
            optimizer_generation_after=1,
            model_sha256_before="0" * 64,
            model_sha256_after="1" * 64,
            optimizer_sha256_before="2" * 64,
            optimizer_sha256_after="3" * 64,
            rng_sha256_before="4" * 64,
            rng_sha256_after="5" * 64,
        )
        before_consumptions = set(bridge._ISSUED_CONSUMPTIONS)

        class FailingPhaseMap(weakref.WeakKeyDictionary):
            def __setitem__(self, key, value):
                raise RuntimeError("persistent phase registry failure")

        original = bridge._TRANSITION_PHASES
        failing = FailingPhaseMap()
        for key, value in original.items():
            weakref.WeakKeyDictionary.__setitem__(failing, key, value)
        bridge._TRANSITION_PHASES = failing
        try:
            bridge._mark_transition_trace_consumed(
                transition,
                receipt_token=object(),
                trace_session_token=object(),
                trace_record_index=0,
            )
        except RuntimeError as error:
            assert "persistent phase" in str(error)
        else:
            raise AssertionError("persistent mark failure was hidden")
        finally:
            bridge._TRANSITION_PHASES = original

        transition_state = bridge._ISSUED_TRANSITIONS[transition]
        session_state = bridge._EXECUTOR_SESSIONS[authority]
        assert transition_state.phase == "FAILED"
        assert transition_state.trace_authority is None
        assert transition_state.trace_consumption is None
        assert session_state.failed
        assert session_state.active_transition_token is None
        assert transition in bridge._FAILED_TRANSITIONS
        assert bridge._TRANSITION_CONSUMPTIONS.get(transition) is None
        assert bridge._TRANSITION_CLAIM_GUARDS.get(transition) is None
        assert set(bridge._ISSUED_CONSUMPTIONS) == before_consumptions
        try:
            bridge._mark_transition_trace_consumed(
                transition,
                receipt_token=object(),
                trace_session_token=object(),
                trace_record_index=0,
            )
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("terminal mark was retried")
        try:
            bridge._executor_session_snapshot(authority)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("terminal mark session remained live")
        """
    )


def test_persistent_accept_lifecycle_registry_failure_is_terminal() -> None:
    _run_isolated_bridge_probe(
        """
        import weakref
        import numpy as np
        import falsewake.experiment_002_training_bridge as bridge

        authority = bridge._issue_registered_executor_session(20_260_720)
        epoch_token = object()
        batch_token = object()
        receipt_token = object()
        transition = bridge._issue_registered_optimizer_transition(
            executor_authority=authority,
            epoch_session_token=epoch_token,
            batch_token=batch_token,
            zero_based_epoch=0,
            zero_based_global_update=0,
            batch_size=bridge._REGISTERED_BATCH_SIZE,
            learning_rate=bridge._registered_learning_rate(0),
            batch_mean_training_loss=np.float32(0.5),
            returned_preclip_l2_norm=np.float32(1.0),
            optimizer_generation_before=0,
            optimizer_generation_after=1,
            model_sha256_before="0" * 64,
            model_sha256_after="1" * 64,
            optimizer_sha256_before="2" * 64,
            optimizer_sha256_after="3" * 64,
            rng_sha256_before="4" * 64,
            rng_sha256_after="5" * 64,
        )
        consumption = bridge._mark_transition_trace_consumed(
            transition,
            receipt_token=receipt_token,
            trace_session_token=object(),
            trace_record_index=0,
        )

        class FailingLifecycleMap(weakref.WeakKeyDictionary):
            def __setitem__(self, key, value):
                raise RuntimeError("persistent lifecycle registry failure")

        original = bridge._CONSUMPTION_LIFECYCLES
        failing = FailingLifecycleMap()
        for key, value in original.items():
            weakref.WeakKeyDictionary.__setitem__(failing, key, value)
        bridge._CONSUMPTION_LIFECYCLES = failing
        try:
            bridge._accept_trace_consumed_transition(
                transition,
                consumption,
                receipt_token=receipt_token,
                executor_authority=authority,
                epoch_session_token=epoch_token,
                batch_token=batch_token,
            )
        except RuntimeError as error:
            assert "persistent lifecycle" in str(error)
        else:
            raise AssertionError("persistent accept failure was hidden")
        finally:
            bridge._CONSUMPTION_LIFECYCLES = original

        transition_state = bridge._ISSUED_TRANSITIONS[transition]
        consumption_state = bridge._ISSUED_CONSUMPTIONS[consumption]
        session_state = bridge._EXECUTOR_SESSIONS[authority]
        assert transition_state.phase == "FAILED"
        assert consumption_state.failed and not consumption_state.used
        assert session_state.failed
        assert session_state.active_transition_token is None
        assert transition in bridge._FAILED_TRANSITIONS
        assert consumption in bridge._FAILED_CONSUMPTIONS
        assert consumption not in bridge._USED_CONSUMPTIONS
        assert transition not in bridge._ACCEPTED_TRANSITIONS
        try:
            bridge._accept_trace_consumed_transition(
                transition,
                consumption,
                receipt_token=receipt_token,
                executor_authority=authority,
                epoch_session_token=epoch_token,
                batch_token=batch_token,
            )
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("terminal accept was retried")
        try:
            bridge._executor_session_snapshot(authority)
        except bridge.Experiment002TrainingBridgeError:
            pass
        else:
            raise AssertionError("terminal accept session remained live")
        """
    )


def test_synthetic_transition_guard_rejects_coherent_rewrite() -> None:
    transition = bridge._issue_synthetic_optimizer_transition(
        seed=7,
        zero_based_epoch=0,
        zero_based_global_update=0,
        batch_size=2,
        learning_rate=0.1,
        batch_mean_training_loss=np.float32(0.5),
        returned_preclip_l2_norm=np.float32(1.0),
        optimizer_generation_before=0,
        optimizer_generation_after=1,
        model_sha256_before=_DIGEST,
        model_sha256_after="1" * 64,
        optimizer_sha256_before="2" * 64,
        optimizer_sha256_after="3" * 64,
        rng_sha256_before="4" * 64,
        rng_sha256_after="5" * 64,
    )
    state = bridge._ISSUED_TRANSITIONS[transition]
    rewritten = replace(state.snapshot, model_sha256_after="6" * 64)
    state.snapshot = rewritten
    state.issuer = replace(
        state.issuer,
        authority_payload=bridge._frame_transition_authority(
            rewritten, route_marker=None
        ),
    )
    with pytest.raises(bridge.Experiment002TrainingBridgeError, match="issuer state"):
        bridge._transition_snapshot(transition)


def test_transition_phase_cannot_be_coherently_reset_after_acceptance() -> None:
    # A subprocess-free synthetic proof exercises the monotonic phase guard:
    # synthetic transitions cannot claim a registered trace, so the failed set
    # itself must defeat a coherent state/map reset.
    transition = bridge._issue_synthetic_optimizer_transition(
        seed=8,
        zero_based_epoch=0,
        zero_based_global_update=0,
        batch_size=1,
        learning_rate=0.1,
        batch_mean_training_loss=np.float32(0.5),
        returned_preclip_l2_norm=np.float32(1.0),
        optimizer_generation_before=0,
        optimizer_generation_after=1,
        model_sha256_before=_DIGEST,
        model_sha256_after="1" * 64,
        optimizer_sha256_before="2" * 64,
        optimizer_sha256_after="3" * 64,
        rng_sha256_before="4" * 64,
        rng_sha256_after="5" * 64,
    )
    with pytest.raises(bridge.Experiment002TrainingBridgeError, match="synthetic"):
        bridge._mark_transition_trace_consumed(
            transition,
            receipt_token=object(),
            trace_session_token=object(),
            trace_record_index=0,
        )
    state = bridge._ISSUED_TRANSITIONS[transition]
    state.phase = "ISSUED"
    bridge._TRANSITION_PHASES[transition] = "ISSUED"
    with pytest.raises(bridge.Experiment002TrainingBridgeError, match="issuer state"):
        bridge._transition_snapshot(transition)
