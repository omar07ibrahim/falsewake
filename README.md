# FalseWake

FalseWake is an experiment in streaming, open-set keyword spotting on ordinary
CPUs. The question is deliberately practical:

> Can a small model recognize ten voice commands without waking up repeatedly
> during unrelated speech?

Most starter examples score isolated one-second clips. A real listener sees a
continuous waveform, overlapping windows, silence, background sound, and speech
that belongs to none of its classes. FalseWake will measure that setting directly:
false accepts per hour, missed commands, detection delay, and CPU cost.

## Current status

Experiment 001 completed two byte-identical development replays and rejected the
registered threshold grid. No threshold satisfies both gates, so the machine-readable
selection is `reject` with `selected_threshold_milli: null`; the held-out
LibriSpeech `test-clean` archive remains absent and unread.

Experiment 002 was registered and launched exactly once as an iterative engineering
response to that failure. The attempt terminated during registered seed execution
and published a canonical `execution_failure`; it admitted no history, checkpoint,
neural score, ONNX artifact, or benchmark. The immediate cause was an
execution-layer race: Linux may omit a process leader's `VmRSS` after releasing its
address space but before the leader becomes a zombie, while the original supervisor
required that record on every sample. The immutable
[outcome](reports/experiment-002-training.json) and
[incident record](reports/experiment-002-execution-incident.json) preserve the
failure and its provenance.

The scientific design remains frozen: a 23,724-parameter causal
depthwise-separable TCN with explicit ONNX state, a 995 ms receptive field, and
deterministic CPU-only training. Experiment 002 will not be retried. A separately
registered Experiment 003 must inherit its data, seeds, model, gates, and runtime
while changing only the diagnosed execution layer. Because experiment 001 already
exposed the complete development curve and top errors, any later replay is
development—not a blind or independent evaluation. The frozen Experiment 002
contract is in [docs/experiment-002.md](docs/experiment-002.md).

At the last threshold that preserves the registered 80% conditional-retention gate
(`0.395`), the listener emits 1,895.7103 false events per scored hour. At the first
threshold that meets the 1.0-event/hour negative gate (`0.991`), it retains only
0.14368% of the baseline's correct decisions. The two frontiers are 0.596 threshold
units apart.

[![Experiment 001 has no feasible threshold on the registered grid](reports/experiment-001-gate-feasibility.svg)](reports/experiment-001-analysis.html)

The portable [analysis](reports/experiment-001-analysis.html), its canonical
[artifact](reports/experiment-001-analysis.artifact.json), full
[1,001-point replay report](reports/experiment-001-dev-replay.json), canonical
[selection](reports/experiment-001-selection.json), and
[reproducibility record](reports/experiment-001-reproducibility.json) preserve the
negative result. Packaging and source-query checks are recorded in the
[analysis verification](reports/experiment-001-analysis-verification.json). The two
replay reports have SHA-256
`b8e30e26498af2600ece01f4cceb436e10a546b8390f039ca8a23cda45f00f2d`; the
selection artifact has SHA-256
`1d44ae6ff06a5fab1567d0342299e293fe001b8c91f9972cfa8e79a2dabf2318`.

The frozen linear floor reaches 56.4% accuracy and 0.558 macro F1 on 4,884 held-out
Speech Commands clips. Its more important failure is open-set: 329 of 405 sampled
`unknown` words are misclassified as one of the ten target commands. The perfect
`silence` recall is much narrower evidence—it comes from overlapping windows in one
held-out noise file, not an independent or continuous-noise benchmark.

The full metrics are in
[reports/experiment-000-linear.json](reports/experiment-000-linear.json), and the
33 KiB portable model is in
[models/experiment-000-linear.json](models/experiment-000-linear.json). Two exact
feature extractions were byte-identical, and an independent refit reproduced every
coefficient and prediction. These are threshold-free clip diagnostics: false
events per hour were measured separately by experiment 001 on development speech,
not on the held-out stream.

![Validation and test open-set clips predicted as a target command](reports/experiment-000-open-set.png)

The hatched bars are validation. The exact numerators and denominators are printed
on the figure; the companion
[per-class recall chart](reports/experiment-000-class-recall.svg) keeps the same
zero-based scale across both splits.

The initial study uses:

- the ten-command Speech Commands v0.02 task (`yes`, `no`, `up`, `down`, `left`,
  `right`, `on`, `off`, `stop`, and `go`);
- the remaining words as open-set speech during training and clip evaluation;
- LibriSpeech `dev-clean` for threshold selection and `test-clean` for the final
  continuous-speech false-accept estimate; and
- a simple log-mel baseline before any neural architecture is introduced.

Dataset roles, licenses, and contamination rules are recorded in
[docs/data.md](docs/data.md). The first experiment and its go/no-go decisions are in
[docs/experiment-000.md](docs/experiment-000.md).
The confidence-threshold continuous replay protocol is registered separately in
[docs/experiment-001.md](docs/experiment-001.md).
The causal neural training, ONNX, and next replay protocol is preregistered in
[docs/experiment-002.md](docs/experiment-002.md).

Before replay, two complete `dev-clean` payload audits independently produced the
same 2,703-row manifest (`SHA-256 6494fa…cff7`) and the same compact
[audit report](reports/dev-clean-audit.json). The audit binds every compressed FLAC,
decoded PCM stream, transcript, and metadata file without extracting the archive.
A fail-closed loader now requires a reviewed, tracked development-selection artifact
before it can acquire `test-clean`. It validates Git through an isolated object-only
snapshot and gives the eventual supplier an anonymous descriptor rather than a
path. That descriptor is irreversibly sealed and matched to the pre-registered
official archive identity before evaluation. The tracked artifact now records
`reject`, so the loader denies holdout access before any supplier can provide bytes.

## What will count as progress

The project will report more than clip accuracy:

- false accepts per hour on continuous negative speech;
- correct-detection recall at a threshold selected only from registered development
  sources;
- detection latency at several audio chunk sizes;
- results for speakers absent from training;
- model size, peak memory, and real-time factor on a named CPU; and
- examples of recurring failure modes, including confusing word pairs.

The planned deployment target is a small ONNX model running locally in a browser
through WebAssembly. A WAV-upload demo comes before live microphone support so the
same audio can be replayed in Python and JavaScript.

## Milestones

1. Audit the speaker split and publish a log-mel linear baseline.
2. Train one compact streaming model and compare three seeds.
3. Calibrate open-set rejection and measure continuous false accepts.
4. Export FP32 and int8 ONNX models with parity and CPU benchmarks.
5. Publish the browser demo, failure gallery, model card, and data card.

FalseWake does not currently claim production wake-word performance, robustness to
every acoustic environment, or a benchmark advantage over existing engines.

## License

Code is released under the [MIT License](LICENSE). Datasets are not redistributed;
their original licenses and required attribution remain in force.
