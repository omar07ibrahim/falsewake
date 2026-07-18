# Experiment 001: replay the linear floor over continuous speech

## Question

Does any confidence threshold on a registered 0.001-spaced grid make the experiment
000 linear classifier useful on continuous, unrelated speech without discarding
most correctly recognized commands?

This experiment is registered after seeing the clip result and before downloading
or inspecting LibriSpeech `dev-clean`. It does not refit the scaler or classifier.

## Negative stream

LibriSpeech `dev-clean` is read English audiobook speech. It is a development source
for threshold diagnostics, not a final test set. The official OpenSLR archive is
`dev-clean.tar.gz` with MD5 `42e2234ba48799c1f50f24a7926300a1` and a published
download size of 337 MB. The downloaded 337,926,286-byte archive has SHA-256
`76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3`.
The audit will reject unsafe or unexpected archive members and require an exact
regular-file union: every canonical split FLAC exactly once, exactly one canonical chapter
transcript for each and only each represented chapter, and exactly the five named
top-level LibriSpeech metadata files in the config. Every other regular member is an
error. For each utterance the audit will bind the raw FLAC, decoded little-endian
PCM16, sample count, canonical numeric ID, and transcript. Chapter transcript bytes
must form an exact bijection with FLAC IDs. The manifest uses ascending relative
POSIX paths and records full-population digests, counts, duration, and decoder
versions. Audio is decoded directly to `int16`; resampling is forbidden.

An utterance is replayed independently. One-second windows start every 100 ms; a
window never crosses an utterance boundary, and a partial tail is dropped. Source
duration and actually scored exposure are reported separately.

Speech containing words such as “yes” remains negative here: narration is not a
direct command to the listener. This experiment measures unintended activations,
not whether a transcript happens to contain a target spelling.

## Scores and events

The portable experiment 000 scaler and linear weights produce 12 logits in
`float64`. Stable softmax subtracts the row maximum before exponentiation. A window
qualifies only when its overall argmax is one of the ten targets and that target's
probability is greater than or equal to the registered threshold.

The first qualifying window emits an event and starts a global one-second refractory
period within that utterance. There is no temporal smoothing, target-specific
cooldown, or post-hoc merging rule. Each threshold has independent state. Its
`next_allowed` sample starts at zero; an event at sample `s` sets it to `s + 16,000`,
and a window starting exactly there may emit. State resets at every utterance
boundary. An exact argmax tie goes to the first portable-model class index.

For an utterance with `N` samples, the full-window count is
`max(0, 1 + floor((N - 16,000) / 1,600))`. Scored exposure is zero when that count
is zero; otherwise it is `16,000 + (count - 1) * 1,600` samples. The corpus rate
uses the sum of those samples divided by 57,600,000 samples per hour.

Thresholds are exactly `float64(i) / float64(1000)` for every integer `i` from 0
through 1,000. Reports and the selection artifact use the integer `threshold_milli`
as the canonical JSON representation; probabilities are derived at runtime. The
experiment can only conclude whether a point on this grid passes;
it does not prove that no real-valued threshold between grid points would pass. The
complete grid and exact event rule are in
[configs/experiment-001.json](../configs/experiment-001.json).

## Positive side of the trade-off

The same thresholds are applied to the ten target classes in the existing Speech
Commands validation sample. A correct accept requires both the correct target
argmax and a probability at or above threshold. Absolute correct-accept recall uses
all 3,703 target clips as its denominator. Conditional retention uses the 2,088
clips whose target argmax is already correct at threshold zero; it measures how many
of the baseline's existing correct decisions survive thresholding. These clips do
not refit anything, and validation remains the only positive source in this phase.

## Decision rule

A threshold passes only if:

- it retains at least 80% of the target clips classified correctly at threshold
  zero; and
- false events on `dev-clean` are at most 1.0 per scored hour.

If several registered thresholds pass, the lowest is selected. If none passes, the
registered grid is rejected, no configuration is selected, and LibriSpeech
`test-clean` stays untouched. The full curve is kept even when the outcome is
negative.

Rates aggregate raw event counts and scored exposure across the corpus. The
selection gate uses that raw count divided by exposure point estimate, not either
confidence interval endpoint. Positive metrics are micro-aggregated and also broken
down by target class. The report gives
both nominal 95% Garwood Poisson intervals and a deterministic 10,000-resample
speaker-cluster bootstrap interval. Speakers are ordered by ascending integer ID.
One `(10,000, speaker_count)` matrix drawn by NumPy `Generator(PCG64(20260718))`
is shared across every threshold, and linear-method percentiles are used. Refractory
events and audiobook speech need not follow a Poisson process, so the Garwood
interval is descriptive, not a guarantee.

The full report keeps aggregate integer sufficient statistics and derived metrics
for all 1,001 grid points. Event-level examples are limited to the 50
highest-confidence events at each unique decision point: zero, the maximum threshold
meeting the retention gate, the minimum threshold meeting the negative gate if one
exists, and the selected threshold if the run passes. This keeps both sides of a
rejection reviewable without turning the report into tens of megabytes of repeated
windows.

## Test firewall contract

Finishing a development replay does not authorize opening `test-clean`. This commit
registers the firewall contract; it does not yet claim that the loader exists. Its
implementation and negative tests are mandatory before acquiring, opening a path
to, or reading any byte from `test-clean`.

A separate selection artifact must bind the experiment config SHA, the exact
seven-file scorer-source bundle, its clean implementation commit, a canonical
runtime-identity digest, dev archive, dev manifest, distinct audit and replay
reports, status, and exact integer `selected_threshold_milli`. Source paths and the
domain-separated length-prefixed digest are fixed in the config; each file must
equal its blob at the recorded `HEAD`. This means a later documentation-only commit
does not invalidate a result. The replay report records exact Python, NumPy, SciPy,
SoundFile, libsndfile, and platform versions. The validator must independently
recompute whether the hash-bound replay report has a
passing grid point from integer counts and samples and, if so, the lowest such
integer point. A hand-written `pass` is not trusted. Before accepting or opening any
test path, the future loader must reject absent, rejected, tampered, dirty,
runtime-mismatched, or identity-mismatched state. Tests must cover each of those
cases and a wrong selected threshold. A
protocol or scorer change starts a new experiment.

Only after a valid selection may the same threshold be evaluated once on both
LibriSpeech `test-clean` and the Speech Commands test target clips. Neither final
source can influence selection.

## What this experiment will not claim

- `dev-clean` is clean read speech, not a complete acoustic-noise benchmark.
- Utterance boundaries are known and reset event state.
- The event rule is a deliberately simple diagnostic, not a production debounce.
- A development-set rate is not the final false-accept estimate.
- No result from `test-clean` exists unless a grid point first passes the registered
  selection rule and produces a matching machine-verified selection artifact.

The experiment still reports absolute correct-accept recall. It cannot exceed the
threshold-zero target accuracy of 56.39%; conditional retention prevents that known
model limitation from making the continuous negative stream irrelevant.

## Status

The protocol and the future firewall contract are registered. The firewall is not
yet implemented. `dev-clean` was downloaded only after the initial registration;
its size and official MD5 were verified and its SHA-256 was recorded. No tar member
or decoded audio has been inspected, and no continuous metric has been calculated.
