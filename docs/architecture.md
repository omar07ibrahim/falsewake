# FalseWake architecture and evidence boundaries

FalseWake is a streaming, open-set keyword-spotting research system with two
distinct bodies of work:

1. Experiments 000 and 001 form a measured baseline pipeline.
2. Experiments 002 through 006 form an implemented causal-model and controlled
   execution lineage with no admitted neural result.

Tested model code, a frozen training plan, and a sophisticated execution
boundary are engineering evidence, not substitutes for a completed benchmark.

## Component contracts

| Layer | Contract | Primary implementation | Evidence boundary |
| --- | --- | --- | --- |
| Corpus audit | Validate archive shape, split membership, speaker identity, audio format, and payload digests | [Speech Commands audit](../src/falsewake/speech_commands.py), [LibriSpeech audit](../src/falsewake/librispeech.py) | Audio remains external and is not redistributed |
| Deterministic frontend | Decode normalized PCM16 and produce fixed 40-bin log-mel frames; summarize them to 80 values for the linear baseline | [features.py](../src/falsewake/features.py) | A deterministic representation is not a quality result |
| Experiment 000 | Build a deterministic sample, materialize a bound feature matrix, fit the fixed linear classifier, and report validation/test metrics | [baseline_data.py](../src/falsewake/baseline_data.py), [feature_matrix.py](../src/falsewake/feature_matrix.py), [linear_baseline.py](../src/falsewake/linear_baseline.py) | This is the measured clip-level floor |
| Experiment 001 | Score continuous development speech, form refractory events, evaluate 1,001 thresholds, and apply both registered gates | [continuous_replay.py](../src/falsewake/continuous_replay.py) | The result is a development-only rejection |
| Holdout firewall | Recompute selection, verify source and Git identities, and call the protected supplier only after a valid passing selection | [holdout.py](../src/falsewake/holdout.py) | The current rejected selection stops before `test-clean` acquisition |
| Causal model | Map `[B, 40, T]` features and explicit state to per-frame logits and next state | [causal_kws.py](../src/falsewake/causal_kws.py) | Implemented and architecture-tested; not a trained benchmark |
| Scientific training core | Freeze data roles, normalization, RNG, populations, optimizer transitions, validation arithmetic, checkpoint ranking, and histories | [Experiment 002 modules](../src/falsewake) and [training protocol](../configs/experiment-002-training.json) | Scientific identities remain Experiment 002 identities across recovery profiles |
| Run authority | Verify the exact registration, committed source bundle, frozen inputs, runtime, and repository topology before issuing an opaque capability | [shared authority engine](../src/falsewake/experiment_002_run_authority.py) and fixed-profile facades such as [profile 006](../src/falsewake/experiment_006_run_authority.py) | Callers cannot select a profile, path, namespace, or retry |
| Runner | Classify the fixed parent/child route and admit child project code only from a sealed source bundle | [Experiment 006 runner](../src/falsewake/experiment_006_runner.py) | Mutable checkout source is not a child fallback |
| Supervisor | Pin launch inputs and enforce process, affinity, RSS, wall-time, output, and cleanup contracts | [Experiment 006 supervisor](../src/falsewake/experiment_006_supervisor.py) | Its trust claim covers the controlled launch surface, not arbitrary code |
| Coordinator and worker | Own one registered lifecycle, supervise three seed assignments plus one selected-seed rerun, and expose no tuning controls | [Experiment 006 coordinator](../src/falsewake/experiment_006_coordinator.py), [seed worker](../src/falsewake/experiment_006_seed_worker.py) | Numerical imports occur only after registered activation |
| Final evidence | Convert verified child frames into deterministic completed-outcome evidence | [Experiment 006 final evidence](../src/falsewake/experiment_006_final_evidence.py) | Tamper-evident within its stated trusted process model; not a security sandbox |
| Final publication | Link immutable payloads in fixed order and link the report last | [Experiment 006 final publication](../src/falsewake/experiment_006_final_publication.py) | A report cannot claim an incomplete payload prefix |
| Safe inspection | Validate a closed set of tracked reports by path, size, SHA-256, and schema, then render human or JSON output | [inspect_evidence.py](../tools/inspect_evidence.py) | No project imports, Git, subprocesses, arbitrary paths, or experiment routes |

## Measured pipeline: Experiments 000 and 001

The measured path is intentionally conventional and reviewable:

```text
Speech Commands audit -> sampling -> log-mel -> 80-value summary
  -> scaler + multinomial linear classifier -> clip metrics
LibriSpeech dev-clean audit -> window scores -> refractory events
  -> 1,001 thresholds -> both gates -> rejected selection
```

Experiment 000 reports 4,884 test clips, 56.4087% accuracy, and 0.557591
macro-F1. The more important open-set observation is that 329 of 405 sampled
`unknown` clips were predicted as one of the ten targets. These values come from
the canonical [metrics report](../reports/experiment-000-linear.json), not from
documentation fixtures.

Experiment 001 replays 2,703 `dev-clean` utterances from 40 speakers, covering
308,310,400 scored samples. The highest threshold that preserves the retention
gate is `0.395`; the first threshold that satisfies the negative-rate gate is
`0.991`. No point on the registered grid satisfies both, so the durable
[selection artifact](../reports/experiment-001-selection.json) records `reject`
and a null selected threshold.

This is development evidence from clean audiobook speech. It is not a
production false-activation estimate, and it does not include a `test-clean`
result.

## Explicit-state causal path

Experiment 002 replaces the 80-value clip summary with a causal network:

![Implemented causal TCN and its explicit streaming state](images/readme/causal-tcn-state.svg)

The registered architecture has a 40-channel input, a 48-channel `1x1` stem,
eight residual depthwise-separable causal blocks with dilations
`1, 2, 4, 8, 1, 2, 4, 8`, per-frame channel normalization, no BatchNorm, a
38-frame causal mean, a 12-class head, and 23,724 trainable parameters.

The TCN receptive field is 61 frontend frames and the pooled model receptive
field is 98 frames. Streaming state contains eight depthwise histories, a
`[B, 48, 37]` pool history, and a saturating `int64` frame counter. Its floating
portion contains 4,656 `float32` values, or 18,624 bytes per batch element.

The interface accepts arbitrary non-empty feature chunks and returns frame
logits plus the next explicit state. Tests cover full-sequence, one-frame, and
irregular-chunk agreement, as well as prefix causality:
[test_causal_kws.py](../tests/test_causal_kws.py).

These are architecture and test-contract properties. No admitted neural
checkpoint exists, so they are not neural accuracy, ONNX parity, latency,
memory, or deployment results.

## Two planes: science and execution

FalseWake separates what should be computed from whether a particular
computation is authorized and publishable.

### Scientific plane

The scientific plane fixes data roles and populations, normalization and PCM
cache identities, model architecture, seeds, augmentation, optimizer and
schedule, training and validation arithmetic, checkpoint/rerun ordering, and
metric gates.

The core is spread across
[data](../src/falsewake/experiment_002_data.py), [preprocessing](../src/falsewake/experiment_002_preprocessing.py),
[RNG](../src/falsewake/experiment_002_rng.py), [training population](../src/falsewake/experiment_002_training_population.py),
[registered executor](../src/falsewake/experiment_002_registered_executor.py),
[registered evaluator](../src/falsewake/experiment_002_registered_evaluator.py),
[metrics](../src/falsewake/experiment_002_metrics.py), and
[registered history](../src/falsewake/experiment_002_registered_history.py).

Recovery profiles 003 through 006 retain those Experiment 002 scientific
inputs and `exp002` sampling, augmentation, history, and model domains.
Changing those domains would change the science, not merely the infrastructure.

### Execution and evidence plane

The outer profile supplies a disjoint registration, runner, transport domain,
scratch/staging roots, managed paths, and publication namespace. Its layers are:

1. **Authority:** verify exact committed inputs and issue one profile-bound
   capability.
2. **Runner:** establish the parent or authenticated sealed-child route.
3. **Supervisor:** launch and account for controlled child processes.
4. **Coordinator:** assign the four fixed child roles and assemble one outcome.
5. **Evidence builder:** independently frame a completed or controlled-failure
   outcome.
6. **Publisher:** create immutable destinations and publish the report last.

The execution plane is fail-closed, but it is not a general security sandbox.
The source explicitly excludes coordinated arbitrary same-process mutation,
`ctypes`, debuggers, and direct memory rewriting from the final-evidence trust
claim. The design should be read as a narrow experimental-integrity contract,
not as isolation from a hostile operating-system user.

## Holdout boundary

The Experiment 001 firewall validates the committed config, scorer sources,
replay report, selection artifact, runtime identity, Git tree/index state, and
official archive identity. Only then can it provide a sealed, read-only
`test-clean` handle to an evaluator.

The current artifact is a valid rejection. The tested behavior is to fail
before invoking the archive supplier:
[test_valid_reject_artifact_never_opens_test](../tests/test_holdout.py).
Consequently, no holdout metric exists.

The planned neural phase does not inherit this authorization. A future neural
development pass would require a new model-specific firewall before any
`test-clean` access, as stated in the
[Experiment 002 protocol](experiment-002.md).

## One-attempt boundary

Experiment 003 demonstrated that a registered invocation can be permanently
consumed before a publisher has enough state to create an automatic report.
Its admission failure therefore did not make the registration reusable.

Profiles 004 onward added a host-persistent parent marker claimed with exclusive
creation before importing the authority or coordinator. A failure after marker
creation is terminal, and the marker is never deleted for retry.

- Experiment 004 and 005 stopped at pre-registration verification, so their
  canonical markers were never created.
- Experiment 006 passed admission, created its permanent marker, and consumed
  its sole registered invocation.

The marker is evidence of attempt consumption. It is not evidence that training
completed, or even that a particular hidden execution stage started.

## Report-last publication boundary

Publication prepares immutable payloads under fixed canonical names. Histories
and an eligible selected model are linked first; the final report is linked
last and is the only completed-outcome commit marker.

If publication fails after linking an initial payload prefix, no final report
can claim that the full set exists. Canonical destinations are not overwritten,
renamed, removed, or made mutable by the publisher.

An absent automatic report still does not authorize retry. Experiments 003 and
006 instead preserve separately classified incident records describing what the
retained evidence can and cannot establish.

## Immutable execution lineage

![Measured results and terminal execution dispositions](images/readme/experiment-lineage.svg)

| Experiment | Final disposition | What is established |
| --- | --- | --- |
| 002 | One registered invocation ended with `seed_selection_failed` | An automatic execution-failure report; no admitted history, metric, artifact, or reusable checkpoint |
| 003 | One registered invocation failed admission on a live route-signature mismatch | No seed child, training process, optimizer update, validation example, automatic report, artifact, or reusable checkpoint |
| 004 | Frozen protocol rejected at pre-registration verification | No registration, invocation, permanent marker, or scientific computation |
| 005 | Frozen facade-normalization recipe rejected at pre-registration verification | No registration, invocation, permanent marker, or scientific computation |
| 006 | One registered invocation ended at final authority verification | Permanent attempt marker and terminal incident; no automatic report, scientific result, artifact, or reusable checkpoint |

Experiment 006's hidden execution progress is not established from retained
evidence. A concurrent `git status --short` can reproduce a relevant metadata
change and is a plausible observer-interference mechanism, but overlap with the
failed reverification was not established. It is not a proven root cause.

The recovery lineage ends at Experiment 006. It is not a queue of jobs to rerun.
The exact records are:

- [Experiment 002 automatic report](../reports/experiment-002-training.json) and
  [incident](../reports/experiment-002-execution-incident.json);
- [Experiment 003 incident](../reports/experiment-003-execution-incident.json);
- [Experiment 004 preflight incident](../reports/experiment-004-preflight-incident.json);
- [Experiment 005 preflight incident](../reports/experiment-005-preflight-incident.json); and
- [Experiment 006 terminal incident](../reports/experiment-006-execution-incident.json).

## Safe evidence readers

The closed reader accepts no repository, report, or output path:

```console
python tools/inspect_evidence.py
python tools/inspect_evidence.py --json
```

It reads exactly ten fixed tracked JSON records through no-follow descriptors,
checks regular-file identity, byte count, SHA-256, stable metadata, and strict
schemas, and writes only its result to stdout. It imports no `falsewake` module,
does not invoke Git or subprocesses, and does not recompute scientific metrics.
Its boundary is tested in
[test_inspect_evidence.py](../tests/test_inspect_evidence.py).

The documentation visual check is also separated from registered execution:

```console
python tools/render_portfolio_visuals.py check
```

It inspects an allowlisted source set as text, AST, and evidence. Its only
project import is the reviewed Experiment 000/001 plot renderer, and it rejects
loading any `falsewake.experiment_*` module. Visual source hashes and non-claims
are recorded in the
[provenance manifest](images/readme/provenance.json).

Neither reader can create an attempt marker or turn documentation verification
into a registered experiment.

## Current, not established, and next

| State | Evidence |
| --- | --- |
| **Current** | Measured 000 clip metrics and 001 development replay; deterministic frontend; tested explicit-state causal architecture; frozen scientific contracts; source-bound execution/evidence code; immutable incidents; hash-pinned inspection and visuals |
| **Not established** | Completed neural training or quality metrics; reusable checkpoint; FP32/INT8 ONNX artifact, parity, or benchmark; neural replay; `test-clean`; production robustness; hidden 006 progress or root cause |
| **Next** | Documentation, evidence inspection, and reproducible visual verification without reopening the terminal lineage |

Experiment 006 must not be retried, and there is no Experiment 007 recovery
profile. Any future scientific study must be separately scoped and
pre-registered outside the closed 002–006 recovery lineage, with new execution
identities and a model-specific holdout firewall before protected evaluation.

## Review map

- Data: [roles and contamination rules](data.md).
- Measured work: [Experiment 000](experiment-000.md) and
  [Experiment 001](experiment-001.md).
- Neural protocol: [Experiment 002](experiment-002.md).
- Recovery record: [003](experiment-003.md), [004](experiment-004.md),
  [005](experiment-005.md), and [006](experiment-006.md).
- Verification: [visual provenance](images/readme/provenance.json) and
  [repository tests](../tests).
