# FalseWake

FalseWake is a streaming, open-set keyword-spotting research system for ordinary
CPUs. It asks a deployment-shaped question that clip benchmarks avoid:

> Can a listener recognize ten commands without repeatedly firing on unrelated
> speech?

The repository covers the whole measurement path: dataset auditing, deterministic
audio features, a portable baseline, stateful causal inference, continuous replay,
threshold selection, holdout isolation, and machine-readable evidence. The current
result is useful but negative: the linear floor cannot meet the registered
false-event and retention gates, and the later neural execution lineage produced no
valid benchmark or reusable model.

## Headline results

### Experiment 000 — the clip-level floor

The checked-in 33 KiB logistic-regression model uses 80 summary statistics from a
fixed 40-bin log-mel frontend. On the frozen Speech Commands v0.02 test sample:

| Metric | Result |
| --- | ---: |
| Test clips | 4,884 |
| Accuracy | 56.41% |
| Macro F1 | 0.558 |
| `unknown` clips predicted as a target command | 329 / 405 (81.23%) |
| `silence` clips predicted as a target command | 0 / 405 |

The silence slice comes from overlapping windows in one noise recording, so it is
not evidence of real-world robustness. The full confusion matrices, per-class
metrics, and lexical slices are in the
[canonical report](reports/experiment-000-linear.json); the portable weights are
in [models/experiment-000-linear.json](models/experiment-000-linear.json).

### Experiment 001 — continuous negative speech

The same model was replayed over 2,703 LibriSpeech `dev-clean` utterances from 40
speakers, representing 5.352611 scored hours. Two complete runs produced
byte-identical 1,001-threshold reports.

| Decision point | False events/hour | Conditional retention |
| --- | ---: | ---: |
| Highest threshold that preserves at least 80% retention (`0.395`) | 1,895.7103 | 80.1724% |
| First threshold below 1 false event/hour (`0.991`) | 0.7473 | 0.14368% |

No threshold satisfies both gates, so the registered decision is `reject` and no
threshold was selected. The `test-clean` holdout was never downloaded or read.
This is a development-set diagnostic, not a final false-accept estimate.

[![No registered threshold satisfies both Experiment 001 gates](reports/experiment-001-gate-feasibility.svg)](reports/experiment-001-analysis.html)

The complete evidence is available as a
[replay report](reports/experiment-001-dev-replay.json),
[selection artifact](reports/experiment-001-selection.json), and
[reproducibility record](reports/experiment-001-reproducibility.json).

## System architecture

```text
Speech Commands v0.02                    LibriSpeech dev-clean
         │                                        │
         ▼                                        ▼
archive + split audit                    archive + PCM audit
         │                                        │
         └──────────────┬─────────────────────────┘
                        ▼
             16 kHz PCM → 40-bin log-mel
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
     80-value clip summary   stateful causal TCN
             │               23,724 parameters
             ▼                     │
    portable linear model          ▼
             └──────────┬── explicit streaming state
                        ▼
          overlapping-window continuous replay
                        ▼
      threshold grid → refractory events → gates
```

The implemented neural candidate is an eight-block, 48-channel causal
depthwise-separable TCN with explicit convolution and pooling histories. Its
interface accepts an arbitrary feature chunk and returns both frame logits and the
next state; tests cover one-frame, irregular-chunk, and full-sequence equivalence.
The receptive field is 995 ms and the model has exactly 23,724 trainable
parameters. It is an architecturally tested candidate, not a benchmarked model.

The event-accounting layer is separated from model scoring. It uses exact integer
accounting for exposure and threshold gates, global refractory state within each
utterance, speaker-grained uncertainty intervals, and deterministic artifact
publication.

## Reproducibility and engineering depth

- The Speech Commands audit identifies 105,829 command clips, split membership,
  speaker identities, and six background recordings without redistributing data.
- Two full feature extractions over 46,254 sampled clips were byte-identical; an
  independent refit reproduced every linear coefficient and prediction.
- Two independent audits of the LibriSpeech development payload produced the same
  2,703-row manifest, including compressed-file and decoded-PCM identities.
- Replay reports are bound to their data, model, configuration, source, and Git
  implementation identities. A source or working-tree change fails closed.
- The holdout loader requires a reviewed selection artifact before it can request
  `test-clean`; the current `reject` artifact keeps that boundary closed.
- CI runs Ruff, strict mypy, and portable adversarial tests for numeric behavior,
  streaming state, artifact schemas, filesystem boundaries, and execution
  protocols.

Dataset roles, licenses, and contamination rules are documented in
[docs/data.md](docs/data.md).

## Experiment record

The repository preserves failed attempts instead of silently replacing them with a
better-looking run.

| Experiment | Outcome | Durable evidence |
| --- | --- | --- |
| 000 | Completed linear clip baseline | [metrics](reports/experiment-000-linear.json) |
| 001 | Completed development replay; threshold grid rejected | [analysis](reports/experiment-001-analysis.html) |
| 002 | Invoked once; terminal execution failure; no neural score, benchmark, or reusable checkpoint | [outcome](reports/experiment-002-training.json), [incident](reports/experiment-002-execution-incident.json) |
| 003 | Invoked once; terminal admission failure before training | [incident](reports/experiment-003-execution-incident.json) |
| 004 | Preflight rejected; never registered or run | [incident](reports/experiment-004-preflight-incident.json) |
| 005 | Preflight rejected; never registered or run | [incident](reports/experiment-005-preflight-incident.json) |
| 006 | Invoked once; terminal authority failure; no scientific result or reusable model | [incident](reports/experiment-006-execution-incident.json) |

Experiments 002–006 form one neural recovery lineage. It is closed: there is no
neural benchmark, ONNX result, or model artifact to claim from it, and no retry is
planned under another experiment number.

## Quickstart

Python 3.12 is required. This path installs the portable analysis stack, rebuilds
the checked-in figures into a temporary directory, and runs representative tests.
It does not invoke any registered neural experiment.

```console
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

mkdir -p /tmp/falsewake-plots
falsewake-plots \
  --replay-report reports/experiment-001-dev-replay.json \
  reports/experiment-000-linear.json \
  /tmp/falsewake-plots

python -m pytest -q -W error -p no:cacheprovider \
  tests/test_features.py \
  tests/test_continuous_replay.py \
  tests/test_result_plots.py
```

The data-backed pipeline is exposed through `falsewake-manifest`,
`falsewake-features`, `falsewake-linear`, `falsewake-librispeech`, and
`falsewake-replay`. Exact dataset preparation and experiment contracts are in
[Experiment 000](docs/experiment-000.md),
[Experiment 001](docs/experiment-001.md), and
[Experiment 002](docs/experiment-002.md).

## Project map

| Path | Purpose |
| --- | --- |
| [`src/falsewake/features.py`](src/falsewake/features.py) | Deterministic 16 kHz log-mel frontend |
| [`src/falsewake/causal_kws.py`](src/falsewake/causal_kws.py) | Explicit-state causal TCN |
| [`src/falsewake/continuous_replay.py`](src/falsewake/continuous_replay.py) | Streaming scores, events, uncertainty, and threshold gates |
| [`src/falsewake/speech_commands.py`](src/falsewake/speech_commands.py) | Source audit and speaker-aware manifest |
| [`src/falsewake/holdout.py`](src/falsewake/holdout.py) | Fail-closed selection and holdout boundary |
| [`reports/`](reports/) | Checked-in metrics, analyses, and incident evidence |
| [`tests/`](tests/) | Numeric, data, streaming, security-boundary, and protocol tests |

FalseWake does not claim production wake-word performance, robustness across
acoustic environments, or an advantage over existing engines.

## License

Code is released under the [MIT License](LICENSE). Datasets are not redistributed;
their original licenses and attribution requirements remain in force.
