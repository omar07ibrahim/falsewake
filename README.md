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
accepts per hour and false reject rate have not been measured.

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

## What will count as progress

The project will report more than clip accuracy:

- false accepts per hour on continuous negative speech;
- false reject rate at a threshold chosen on validation data only;
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
