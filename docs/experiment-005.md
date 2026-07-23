# Experiment 005: preflight boundary

## Status

Experiment 005 reached its offline pre-registration boundary and stopped there.
Its frozen execution protocol was introduced in commit
`0b6bf2cac2d4f6a04d6ced596bf8f835660eb072`; the committed
[`configs/experiment-005-execution.json`](../configs/experiment-005-execution.json)
has SHA-256
`824d1677cf8f75567cf1915381f0a9bbc778a74cba879ace80733c3ceb48e54d`.
The implementation candidate is commit
`04b634451d129faadadd921215123547e3467d65`.
The observational protocol proof was added in its direct child
`1e20d6cf210a2d4a59cfc737a14eaccfc94a5b74`; the proof file
`tests/test_experiment_005_protocol.py` has SHA-256
`cf6d4ee2fcba31de37774b591571abb2d86998cc0f2a3e99a40abe1af4af5414`.

This boundary is not a registered invocation. No Experiment 005 registration was
created, and its authority issuer, coordinator, and runner were not invoked. The
permanent Experiment 005 attempt marker was not created. No training, optimizer
update, validation example, model, history, report, or other managed output was
produced. This document is not an incident report and does not authorize an
execution.

## Frozen normalization defect

The correct Experiment 005 facade describes its retained state honestly:

```python
"""Return shared retained state only when its exact profile is ``005``."""
```

Its SHA-256 is
`87d18e901b4b814ce31c7c17b60b6aa54f837edc84ad3bb5ee1af386df006f5e`.
After applying all twelve ordered substitutions frozen in the protocol, its
normalized SHA-256 is
`9778140580667e95e120a4a8472d8b79ffb7034c3886c208fc3fd3c4da966c86`.
The immutable Experiment 004 reference has SHA-256
`cacab13729f14bfdce10afa5b53b2a09116cf56abe21aee12ac9220a6ad2e5b5`.

The normalized candidate differs from that reference by exactly one deleted line
and one added line:

```diff
-    """Return shared retained state only when its exact profile is ``004``."""
+    """Return shared retained state only when its exact profile is ``005``."""
```

The recipe replaces `profile ``005``` with `profile ``004```, but the honest
sentence contains `profile is ``005```. The intervening `is ` means the frozen
substring does not occur. None of the other substitutions can change this line.
Keeping the stale value ``004`` would make the prose byte-compatible but false;
it is not an acceptable implementation repair.

## Recovery

The defect belongs to the already frozen Experiment 005 protocol, so neither the
protocol nor the facade may be rewritten to force admission. Experiment 005 must
remain unregistered and unrun. Any recovery must preserve this protocol and
implementation as evidence, use a new Experiment 006 identifier and namespace,
and freeze a corrected normalization or semantic facade contract before any new
implementation or registered route is considered.
