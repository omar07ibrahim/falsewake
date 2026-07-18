# Experiment 000: establish the floor

## Question

How badly does a straightforward clip classifier behave when its scores are replayed
over continuous unrelated speech?

## Baseline

A multinomial linear classifier receives fixed log-mel summary features from a
one-second, 16 kHz mono window. It predicts the ten target commands plus `unknown` and
`silence`. This is a diagnostic baseline, not the intended streaming model.

## Split and tuning rule

- Train only on the Speech Commands training partition.
- Choose feature normalization, regularization, and the acceptance threshold using
  Speech Commands validation plus LibriSpeech `dev-clean`.
- Evaluate clip classification on the Speech Commands test partition.
- Evaluate continuous false accepts once on LibriSpeech `test-clean` after selecting
  a configuration.
- Keep all utterances from a speaker in one partition.

## Reported numbers

- macro F1 and per-class recall for target commands;
- false reject rate at the selected threshold;
- false accepts per hour on validation and held-out continuous speech;
- the number of evaluated negative hours and 95% Poisson confidence intervals;
- feature extraction and classifier latency on the project host; and
- a confusion matrix plus the highest-scoring false accepts.

## Decisions made before seeing results

- If the linear model cannot separate target clips at all, inspect the feature
  implementation before adding a neural model.
- If clip metrics look good but continuous false accepts are poor, keep that negative
  result as the motivation for the streaming model.
- Do not add architectures until the same waveform produces the same feature tensor
  under every supported chunking pattern.

## Results

The source-data audit is complete; the classifier has not been run yet. The audit
report records 105,829 usable command clips, six partitioned background recordings,
and a byte-stable payload-bound manifest. No model metric has been calculated, and
LibriSpeech has not been inspected.
