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
except the choice of one point on its threshold grid. The `test-clean` stream remains
untouched until one configuration has been selected and the implemented access
firewall passes. Its official size, MD5, and SHA-256 were registered from public
checksum metadata before archive access. A result that changes the configuration
after viewing `test-clean` starts a new experiment rather than replacing the old one.

After that registration and a header-only inspection, two full `dev-clean` audits
decoded all 2,703 FLAC members without filesystem extraction. They produced
byte-identical manifests and reports. The manifest remains a generated local
artifact; its digest and the four domain-separated inventory digests are recorded in
[reports/dev-clean-audit.json](../reports/dev-clean-audit.json).

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
