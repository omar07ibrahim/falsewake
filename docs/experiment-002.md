# Experiment 002: a causal streaming keyword spotter

## Why this experiment exists

Experiment 001 showed that a clip-level linear model cannot trade confidence for a
usable continuous false-activation rate. Experiment 002 therefore replaces the
representation, not just the threshold: a small causal convolutional network keeps
short acoustic history and produces a score while audio is arriving.

This is honest iterative development after the complete Experiment 001
`dev-clean` result. It is not a blind or confirmatory result, and `dev-clean` is not
a holdout for this experiment. This phase-1 protocol is committed before the
Experiment 002 model implementation commit, training, export, and any Experiment
002 continuous development score. A separate replay configuration will be frozen
after deployment selection and before the first new `dev-clean` replay.

The machine-readable contract is
[configs/experiment-002-training.json](../configs/experiment-002-training.json).
It binds the Experiment 001 config, seven-file scorer digest, linear model, replay
report, and rejected selection artifact so the predecessor cannot silently change.

## Data boundaries

Training and validation use the already audited Speech Commands v0.02 archive and
manifest. No LibriSpeech payload may be used for training, normalization,
augmentation, calibration, export, or benchmarking. Speech Commands test audio is
also forbidden throughout this development phase.

The protocol binds the archive and manifest SHA-256 values, the decoded inventory
SHA-256, and the exact validation/testing-list SHA-256 values. Those identities are
checked independently; a matching filename or directory layout is not sufficient.

Each training epoch contains 40,027 examples: all 30,769 target clips, exactly
6,172 unknown clips balanced without replacement across the 25 non-target words,
and 3,086 unique one-second windows from the four training background recordings.
The validation population is fixed at 10,583 examples: all 3,703 target clips, all
6,278 unknown clips, and all 602 full windows from `running_tap.wav` at a 1,600
sample hop. Validation is only for checkpoint and seed selection and registered
metrics. It cannot supply gradients, normalization statistics, calibration
examples, augmentation noise, or fitted LayerNorm state.

Global normalization uses every one of the 84,843 unaugmented training command
clips. Per-mel sums and sums of squares are accumulated in a fixed manifest/frame
order in `float64`; the 40 means and 40 population standard deviations are frozen
as little-endian `float32` values.

## Reproducing the neural toolchain

Experiment 002 separates training, export, and inference dependencies so a runtime
installation need not carry PyTorch or ONNX export tooling. The registered run uses
Python 3.12.3 and the exact direct dependency versions below. On a CPU machine,
create an isolated environment from the repository root and install PyTorch from
its official CPU wheel index before installing the project extras:

```console
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.13.0+cpu"
python -m pip install "numpy==2.5.1" "onnx==1.22.0" "onnxscript==0.7.1" \
  "onnxruntime==1.27.0" "safetensors==0.8.0"
python -m pip install -e ".[dev,train,export,runtime]"
```

The `+cpu` build tag is deliberately recorded in the experiment contract and the
reproduction command rather than imposed as a portable project requirement. The
bounded extras describe compatible environments; the exact registered versions
remain the authority for Experiment 002 evidence.

## Deterministic augmentation and training

Sampling, ordering, and augmentation are stateless HMAC-SHA256 functions of the
registered seed, epoch, example identity, decision domain, and counter. There is no
global data RNG. The byte encodings for command, source-word, and background-window
identities and every ASCII decision domain are enumerated in the machine contract.
Commands receive a zero-filled shift of at most 1,600 samples,
gain in `[-6, 6)` dB, and with probability 0.8 a training-background segment mixed
at an SNR in `[0, 20)` dB. Values are clamped before the fixed log-mel frontend.
One time mask of width 0 through 10 and one mel mask of width 0 through 4 are then
filled with normalized zero. Silence starts from a selected training-background
segment and receives the same gain and feature masking, but no shift or second
noise mix.

Three registered seeds each train from scratch for 30 epochs on CPU `float32`.
There are exactly 313 batches per epoch and 9,390 optimizer updates. AdamW, label
smoothing, gradient clipping, one linear warmup epoch, and the remaining cosine
schedule are fixed in the config. AMP, early stopping, MKLDNN, data-loader workers,
and nondeterministic Torch algorithms are disabled. CPU, memory, scratch-disk, and
wall-time budgets bound cache construction, each training run, and export.

Every independent clip starts from zero neural state and computes all 98 dense
logits, fixing dropout RNG consumption. Cross-entropy supervises only frame 97;
startup frames 0 through 96 never enter the loss. There are no class weights, and
one AdamW parameter group contains every trainable weight, LayerNorm parameter, and
bias.

Within a seed, the checkpoint maximizes 12-class validation macro-F1, then minimizes
validation cross-entropy, then prefers the earlier epoch. The same ordering selects
between seed winners, with the lower seed as the final tie-break. The winner is run
again from scratch and must reproduce the exact canonical history, tensor, and
prediction digests. A mismatch or failed gate ends Experiment 002; it does not
invite a quiet retune under the same ID.

Before any continuous replay, the winner must achieve at least 85% target accuracy,
70% recall for every target, and 80% 12-class macro-F1. No more than 20% of unknown
clips or 5% of silence windows may have a target argmax, and all metrics, logits,
parameters, and states must be finite.

## Causal model contract

Input is normalized log-mel audio in `[B, 40, T]` layout. A 48-channel stem feeds
eight residual depthwise-separable causal blocks with dilations
`1, 2, 4, 8, 1, 2, 4, 8`. Every normalization is per frame across channels; there
is no BatchNorm. A fixed mean of the current and previous 37 encoder frames feeds a
12-class linear layer. Dropout 0.10 exists only between that pool and the linear
layer during training.

The network has exactly 23,724 parameters. Its output receptive field is 98 frontend
frames, or 15,920 samples (995 ms), not one second. The explicit streaming state is
eight causal histories, a `[B, 48, 37]` pool history, and a saturating `int64`
frame counter. Float state occupies 18,624 bytes per batch element. An utterance is
streamed once; state resets only at its boundary. Frame 97 is the first eligible
score, followed by every tenth frame.

The ONNX ABI fixes ordered names and shapes for `mel_frames`, eight depthwise
histories, `pool_state`, and `frames_seen`, plus matching `next_*` outputs. Only the
batch and incoming-frame axes are dynamic; histories are oldest-to-newest. Dense
inference costs `21,504 × F + 576 × S` neural MACs, excluding normalization,
activation, pooling, softmax, and the frontend. Thus a dense 98-frame call is
2,163,840 MACs and a dense ten-frame steady call is 220,800 MACs.

## Portable deployment

The FP32 ONNX export uses opset 18 and explicit state tensors. PyTorch and ONNX
Runtime CPU must agree within `1e-5` on logits and state, with exact argmax on all
10,583 validation examples. Full-sequence, one-frame, irregular-partition, and
append-future prefix-invariance checks protect the causal contract. Dropout,
BatchNorm, and random ONNX nodes are forbidden; the model must fit within 256 KiB.

One static mixed QDQ S8S8 recipe may quantize per-channel Conv, Gemm, and MatMul
weights while leaving LayerNorm, state, and pooling in FP32. Its 1,536 calibration
examples are selected only from exact training universes, 128 per class, by the
registered HMAC rank. They are unaugmented, globally normalized, and presented with
zero explicit state. The complete ONNX Runtime 1.27 quantizer argument set is fixed;
there is no second recipe after seeing its result. INT8 must independently pass the
absolute validation gates, stateful partition/reset parity, registered relative
agreement and probability limits, open-set limits, and the 192 KiB size limit. It
is selected only if eligible and no slower than FP32 at steady-state p95; otherwise
FP32 is selected if FP32 itself passes every gate.

FP32 and any INT8 candidate that was successfully produced are benchmarked on the
EPYC 7R13 with ONNX Runtime CPU in sequential 1/1-thread mode after 200 warmups and
over 5,000 measured iterations. HMAC-derived
feature and PCM inputs are byte-identical between candidates. The 98-frame case is
state-cold but session-warm; the ten-frame case starts from a fixed nonzero state.
Timing uses `perf_counter_ns`, materializes outputs, and reports higher-method
percentiles from a fresh process per candidate. The selected deployment must keep
steady ten-frame model p95 at or below 2 ms,
frontend-plus-model p95 at or below 10 ms, and peak RSS at or below 256 MiB.

## Later continuous replay

The later replay keeps the 0.001 threshold grid and `float64` softmax, but freezes a
more realistic SAME-TARGET 3-of-5 vote: the current tick must qualify and the same
target must qualify on at least three of the last five ticks. A global ten-tick
minimum event spacing permits the next event at tick `k + 10`. History and
refractory state reset at each utterance.
An unsmoothed curve is diagnostic only and can never select a threshold.

Positive validation streams contain one second of zeros, the raw clip, then one
second of zeros. A correct event must be strictly after command onset and no later
than the clip duration plus 8,000 samples after onset. The lowest threshold must
simultaneously achieve no more than one `dev-clean` false event/hour, at least 75%
absolute correct-detection recall, 80% conditional retention, 60% recall for every
target, higher-method p95 latency no greater than 12,800 samples, and target-event
rates no greater than 5% on both validation unknown and silence streams.

Target and unknown streams use one second of zeros, the actual decoded raw clip,
then one second of zeros; silence uses the exact registered 16,000-sample
`running_tap.wav` window. The first class-order index wins exact argmax ties. A zero
or non-finite denominator or latency statistic fails a threshold rather than being
silently skipped.
Every frontend value, ONNX logit, floating state, and softmax probability must also
be finite. A single violation rejects the complete replay rather than suppressing
an event through a NaN comparison.

Two full replays must produce byte-identical evidence. Even a pass does not authorize
the official LibriSpeech `test-clean` archive: it remains absent and unread, and a
new Experiment 002-specific fail-closed firewall would be required before access.
Before either deployment candidate can score `dev-clean`, the separate phase-2
config must bind the phase-1 config, training report, selected weights and tensor
digest, normalization, FP32 ONNX artifact, the INT8 attempt report and its artifact
when one exists, selected representation, export inputs, source bundle and
implementation commit, runtime, benchmark, Speech Commands manifest, and the
registered development archive/manifest/audit SHA-256 values.

## Limitations

Speech Commands is a small controlled-vocabulary corpus, its validation background
comes from one recording, and clean audiobook speech does not cover real devices,
rooms, music, or adversarial speech. Seed and checkpoint selection use the same
fixed validation population that supplies pre-replay gates. The future `dev-clean`
curve is explicitly development evidence already seen at the project level. The
experiment can establish reproducibility and performance on those registered
populations; it cannot turn them into an unseen production estimate.

## Status

Only the phase-1 training and deployment protocol is registered here. Model code,
training results, exports, benchmarks, the separate replay config, and Experiment
002 continuous scores do not yet belong to this phase.
