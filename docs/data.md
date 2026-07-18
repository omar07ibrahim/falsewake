# Data plan

FalseWake keeps downloaded audio outside the repository. Generated manifests contain
relative source identifiers and experiment splits, not copied recordings.

## Speech Commands v0.02

- Role: training, validation, and clip-level testing.
- Source: [TensorFlow download](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz)
- Documentation: [TensorFlow Datasets catalog](https://www.tensorflow.org/datasets/catalog/speech_commands)
- License: Creative Commons Attribution 4.0.
- Download size: approximately 2.37 GiB.
- Archive SHA-256: `af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58`.

The dataset's `validation_list.txt` and `testing_list.txt` define the official
speaker-aware partitions. The manifest builder pins both files, derives speaker IDs
independently, and rejects a source if any speaker crosses partitions. It also hashes
every WAV payload; matching names and headers are not enough to pass the audit.

The first audit completed on 2026-07-18. It found 84,843 training clips, 9,981
validation clips, and 11,005 test clips. The 2,618 derived speakers were disjoint
across those partitions. Two payload-bound builds produced byte-identical JSONL
output; the counts and digests are kept in
[reports/speech-commands-audit.json](../reports/speech-commands-audit.json).

The ten standard commands are target classes. Other recorded words are `unknown`.
The six long background recordings have a file-level split: four for training,
`running_tap.wav` for validation, and `white_noise.wav` for test. A window never
crosses those boundaries. This small, source-specific validation/test background set
is a limitation, not a general acoustic-noise benchmark. Unknown and silence
sampling rates belong to the experiment configuration and will be reported with
every result.

## LibriSpeech

- Role: continuous negative speech only; it is never a positive command source.
- Source: [OpenSLR SLR12](https://www.openslr.org/12/)
- License: Creative Commons Attribution 4.0.
- Validation stream: `dev-clean` (337 MiB archive).
- Held-out test stream: `test-clean` (346 MiB archive).

Every continuous experiment must register its threshold grid, event rule, cooldown,
and smoothing policy before inspecting `dev-clean`. Experiment 001 fixes everything
except the choice of one point on its threshold grid. Two byte-identical development
replays found no point satisfying both registered gates, so the canonical selection
is `reject` with a null threshold. The official `test-clean` archive remains absent
and unread, and the implemented firewall denies access before invoking an archive
supplier. Its official size, MD5, and SHA-256 were registered from public checksum
metadata without archive access. A future experiment that changes the configuration
must register that change rather than replacing this result.

After that registration and a header-only inspection, two full `dev-clean` audits
decoded all 2,703 FLAC members without filesystem extraction. They produced
byte-identical manifests and reports. The manifest remains a generated local
artifact; its digest and the four domain-separated inventory digests are recorded in
[reports/dev-clean-audit.json](../reports/dev-clean-audit.json).

The development replay scored 2,703 utterances from 40 speakers, totaling 5.352611
hours of full-window exposure. Its complete 1,001-point
[report](../reports/experiment-001-dev-replay.json), canonical
[selection](../reports/experiment-001-selection.json), portable
[analysis](../reports/experiment-001-analysis.html), and
[reproducibility record](../reports/experiment-001-reproducibility.json) are retained
even though the grid was rejected. These development outputs do not contain or
summarize any `test-clean` observation.

## Experiment 002 data roles

Experiment 002 is an iterative response to the published linear failure, not a new
blind benchmark. The complete experiment 001 `dev-clean` curve and top-event
transcripts were already available when its neural architecture was registered.
Accordingly, `dev-clean` remains a development source, and an improvement there
will not be presented as an independent generalization estimate.

Only Speech Commands `train` payloads may contribute gradients, normalization
statistics, augmentation noise, or INT8 calibration examples. Speech Commands
`validation` selects checkpoints, the deployment representation, and the positive
side of the future replay threshold. Speech Commands `test` audio is forbidden to
experiment 002 training, export selection, and development replay code. The exact
roles and byte identities are frozen in
[configs/experiment-002-training.json](../configs/experiment-002-training.json).

No LibriSpeech payload is a machine input to neural training or quantization. A
second replay registration must bind the chosen model and implementation before the
first experiment 002 score is computed on `dev-clean`. Even a development `pass`
will not authorize `test-clean`: a new, experiment-specific fail-closed firewall
must be implemented and reviewed first. The official archive remains absent and
unread.

## Attribution

Results using Speech Commands will cite Pete Warden, *Speech Commands: A Dataset for
Limited-Vocabulary Speech Recognition* (2018). Results using LibriSpeech will cite
Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur, *LibriSpeech: An
ASR Corpus Based on Public Domain Audio Books* (2015).

The future browser demo will ship only original code, model weights trained by this
project, aggregate results, and a few recordings contributed specifically for the
demo. It will not bundle either source corpus.

The audit runs on an exclusively owned extraction and output directory. It rejects
symlinks and unexpected inventories, but it is not intended to defend against a
same-user process mutating those directories concurrently.
