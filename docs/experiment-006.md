# Experiment 006: terminal record of the semantic recovery attempt

## Executive summary

Experiment 006 repaired the unsatisfiable facade-normalization contract that
stopped Experiment 005 before registration. The replacement used an exact
semantic AST contract, truthful docstring checks, inverse restoration, a
repository-history oracle, and cross-profile capability rejection. That work
passed admission in commit
`438651010c4ef1de4012a570f3211b1d27bb1e3a`.

The direct-child registration commit
`b73ecb8be3871092aad431a6c497771296c67d60` then authorized one invocation.
That invocation claimed the permanent Experiment 006 marker, invoked the
registered issuer, coordinator, and runner, and exited with status 1 after the
final registration capability was rejected. The terminal incident is preserved
in commit `fd6b98e6122d2884800b359a61440e9b07832204`.

No automatic terminal report, model, reusable checkpoint, or scientific result
was published. Cleanup happened before the failing final authority check, so the
retained evidence does not establish how far any hidden primary execution
progressed. The exact root cause is also unknown. A concurrent `git status
--short` is a plausible source of observer interference, but overlap with the
failing reverification was not established.

The sole Experiment 006 attempt is consumed. There is no retry, Experiment 007,
checkpoint reuse, or further recovery profile in this lineage.

## Immutable commit and artifact record

The authoritative topology is:

1. execution protocol:
   `4adf524f9a3007d5382a851b3419e697d80ceeef`;
2. final admission:
   `438651010c4ef1de4012a570f3211b1d27bb1e3a`;
3. direct-child registration:
   `b73ecb8be3871092aad431a6c497771296c67d60`; and
4. direct-child terminal incident:
   `fd6b98e6122d2884800b359a61440e9b07832204`.

The frozen
[`configs/experiment-006-execution.json`](../configs/experiment-006-execution.json)
has mode `100644`, 129,719 canonical ASCII JSON bytes, and SHA-256
`47ce6c6f11c3575944ef8c4ab1edcc0b42fdfa9bf1ede0318bfc5ea61fabf650`.

The registration commit's sole change added
[`configs/experiment-006-run.json`](../configs/experiment-006-run.json). The
file has mode `100644`, 14,811 canonical ASCII JSON bytes, Git blob
`02e087bbbbeeab87904a3ef5c5fd9bd7acc2985b`, and SHA-256
`53101c6424db09da01e23af4dc5956a57aa8a7e41226113782d4da1d822ab0b8`.
It binds the admission commit, source-bundle SHA-256
`8150d34fa0db80efd073e78c740d0291051dadbafac3eda81cc3534d7c604e95`,
and runtime fingerprint SHA-256
`da1b542e1b6fe8adb6ba56d7060e35e06e39c85dacdf3c35474e404126bff5ea`.

The incident commit's sole change added
[`reports/experiment-006-execution-incident.json`](../reports/experiment-006-execution-incident.json).
The record has mode `100644`, 8,839 canonical JSON bytes, and SHA-256
`c5cf002ea876890e3b3769af2bcc850433d3d08ac1071c864ed69ee2172af77d`.
It is a manual post-exit reconstruction, not an automatic training report. Raw
stderr bytes were not retained, so the record does not claim a raw-stderr hash.

## Why Experiment 006 existed

Experiment 005 ended before registration. Its frozen facade-normalization recipe
could not express the truthful profile-005 `_verified_state` docstring. The
candidate correctly said:

```python
"""Return shared retained state only when its exact profile is ``005``."""
```

The protocol only replaced the substring `profile ``005```; the intervening
`is ` made that replacement inapplicable. Leaving the stale profile `004` would
have satisfied byte parity only by making the documentation false. Rewriting a
frozen contract after seeing its implementation would instead have erased the
failed prediction.

The immutable
[`reports/experiment-005-preflight-incident.json`](../reports/experiment-005-preflight-incident.json),
SHA-256
`3de6fbbd6912bee47f3ef01f5371b57ae8d5942b6fc9915fc0ec243b9e931e3d`,
records that rejection in commit
`594e2c491c057394f18860c8362a51190460645f`. Experiment 006 preserved that
terminal predecessor byte-for-byte and introduced a new authority profile,
execution namespace, registration contract, transport domain, and permanent
one-attempt boundary.

## Intended scientific inheritance

The registered Experiment 006 plan did not retune after the Experiment 005
preflight rejection. It inherited the Experiment 002 scientific protocol: the
same data split, three training seeds, augmentation, normalization,
23,724-parameter causal model, optimizer, schedule, ranking rule, selected-seed
rerun, and gates.

In particular, the registered plan fixed:

- seeds `20260719`, `20260720`, and `20260721`;
- 30 epochs, 313 updates per epoch, and 9,390 updates per seed;
- 40,027 training examples per epoch and 10,583 validation examples;
- 53 parameter tensors and 23,724 parameter values; and
- Python 3.12.3, NumPy 2.5.1, PyTorch 2.13.0+cpu, and safetensors 0.8.0.

The Speech Commands archive, exact 105,829-record manifest, and
2,995,776,948-byte PCM cache remained external inputs. Their required SHA-256
values were respectively
`af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58`,
`d28e6993101bd6bc452033bcb7355b25e3b84ab5c51a1458cc097dc93c60f78b`,
and
`b56270a4b99d235c62c162545b139a79b7cf3dc157259a3bbf5678d8a451f653`.
The `exp002` inner domains continued to define scientific sampling,
augmentation, histories, and model bytes; outer authority, transport, and
publication evidence used profile `006`.

These are properties of the admitted and registered plan, not reported results.
The terminal attempt produced no scientific result from which model quality,
completed updates, validation coverage, or reproducibility could be claimed.

## Semantic admission contract

The failed Experiment 005 recipe treated a small authority facade as text to be
normalized into its predecessor. Experiment 006 instead specified the interface
it needed.

The frozen `semantic_ast_v1` projection parses Python 3.12 syntax, excludes
source locations and only the leading module and function docstring nodes, and
preserves statement order. It requires exact imports, aliases, exports, eight
synchronous non-generator routes, signatures, direct-call bodies, and a closed
top-level surface. Defaults, wrappers, async functions, generators, additional
statements, module-level dynamic `__getattr__`, and exception wrappers are
forbidden.

Docstrings are verified separately. Eight profile-bearing scopes must name
`006` truthfully, while `_required_child_bundle_seals` has exact
profile-neutral wording. Stale claims for profiles `002` through `005` are
rejected. Changing a truthful docstring to `005` therefore leaves the semantic
AST projection unchanged but still fails admission.

Before the protocol was frozen, a synthetic witness satisfied ten positive
checks. Twenty-six one-at-a-time negative mutations demonstrated rejection of
signature drift, wrong-profile delegation, extra wrapper logic, async or
generator conversion, stale documentation, omitted dispatches, unexpected Git
operations, swallowed history faults, self-reference, and early registration
topology. Those operators and expected outcomes were frozen before the
implementation was admitted.

## Shared authority and history proof

The shared-authority change was bounded in both directions. Its positive
contract required exactly 17 profile-006 symbols: ten constant assignments, six
simple fixed-profile functions, and one implementation-history verifier. It
allowed insertions in exactly six existing generic dispatches:

1. `_AUTHORITY_PROFILES`;
2. `_expected_frozen_bindings`;
3. `_frozen_file_bindings`;
4. `_require_authority_profile`;
5. `_require_invocation_registration`; and
6. `_verify_committed_repository`.

No seventh dispatch or arbitrary logic inside an admitted dispatch was allowed.
The inverse contract removed the enumerated additions and restored the terminal
Experiment 005 shared-authority blob exactly as both bytes and semantic AST.

The implementation-history verifier was exercised through an isolated fake-Git
oracle rather than accepted by source appearance. Its AST call closure mapped 13
ordered logical history events to exactly 70 intercepted primitive calls
through `_git`, `_git_process`, `_committed_blob`, and the recursive Experiment
005 history verifier. The oracle bound event order, arguments, results, P5
recursive bindings, source delta, file modes, ancestry, and absence of transient
P4/P5/P6 managed outputs over the historical admission interval. Injecting a
fault into every required event, or adding, removing, duplicating, swapping,
swallowing, or failing to map calls, had to reject.

The pre-registration live-boundary proof imported the seven actual Experiment
006 modules in an isolated process without invoking the registered issuer,
coordinator, or runner. It verified 15 parent routes, four wrapper routes, and
all 20 directed cross-profile rejection cases across five authority profiles.
This was an admission test; it is distinct from the later registered invocation.

## Historical admission record

The final pre-registration gate completed before the registration commit
existed. At that historical boundary, the run config, Experiment 006 marker,
model, histories, and terminal report were absent.

The recorded admission checks were:

- protocol-proof commit:
  `fe21f22feb4d258f291c4d8c8bfc9646f2618227`;
- admitted
  [`tests/test_experiment_006_protocol.py`](../tests/test_experiment_006_protocol.py)
  blob SHA-256:
  `7f4510ff9786db856a5222a7455a75db9a23c603fa39aecf811c65a942cfd2f4`;
- P6-targeted suite: 343 passed with warnings treated as errors;
- portable suite: 3,012 passed with warnings treated as errors;
- host-bound suite: 525 passed with warnings treated as errors;
- strict mypy: no issues in 157 source files;
- repository Ruff lint: passed;
- scoped Ruff formatting: passed for all 19 Python files added or modified since
  frozen base `594e2c491c057394f18860c8362a51190460645f`; and
- actionlint 1.7.12: passed for `.github/workflows/ci.yml` after verifying
  official Linux amd64 release checksum
  `8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8`.

The protocol required formatting but did not define a repository-wide command
or scope. The gate therefore checked the admitted Python delta from its exact
frozen base. Repository-wide formatting was not claimed.

The admission document itself was the sole addition in commit
`438651010c4ef1de4012a570f3211b1d27bb1e3a`, whose parent was the protocol-proof
commit. The direct-child registration then bound that admission commit without
requiring a self-reference in the admission document.

## Consumed attempt and terminal failure

The parent bootstrap claimed
`/home/ubuntu/gitcode/.t/falsewake-experiment-006-attempt` through the frozen
exclusive-creation protocol before activating the registered source tree. The
retained marker has:

- payload `falsewake-experiment-006-attempt-v1\n`;
- 36 bytes;
- mode `0444`;
- SHA-256
  `11cb9ce51db5b5153d3839ac1edad1cd63ae582009417abfea6a8562e319bfbe`;
  and
- mtime `2026-07-24T01:48:08.576998118Z`.

The marker is the permanent proof that the one registered attempt was consumed.
It must not be deleted, changed, or reused.

The process later exited with status 1. The retained exception chain reached the
final registration-frame verification and reported:

```text
Experiment002RunAuthorityError:
run-registration capability was not issued by this verifier

Experiment006CoordinatorError:
registered experiment final authority failed closed
```

The exact capability class had passed the final verifier's type check, and the
same capability had previously passed coordinator admission. The observed
failure therefore occurred in the later compound provenance check, not at an
independent-module class mismatch. The final wrapper did not retain any earlier
primary exception that the coordinator may have held, so the incident record
does not infer one.

After exit, the registration config remained present while the selected model,
three seed histories, selected-seed rerun history, and automatic training report
were absent. Scratch, staging, and publication-temporary paths were also absent.
Because the coordinator cleans up before its final authority check, those
post-exit absences do not establish whether seed children or training processes
started, how many optimizer updates occurred, or whether any validation examples
were evaluated. Each remains not established from retained evidence.

## Root-cause boundary

The root cause is undetermined. While the registered parent was observed alive,
the automation monitor ran `git status --short`. A post-exit check, performed
without calling registered routes, reproduced the relevant mechanism: ordinary
Git status opened `.git/index` read-only, created `.git/index.lock` with
exclusive creation, and then unlinked the lock. The `.git` directory metadata
changed while `.git/index` metadata did not.

The authority's raw-worktree inventory requires a complete `.git` stat frame to
remain unchanged. It is therefore plausible that an overlapping status command
caused reverification to fail and poisoned the capability before the final
check. The retained evidence does not establish that temporal overlap, so this
is a plausible observer-interference hypothesis, not a confirmed cause.

The operational lesson is narrower than a causal claim: after a permanent
attempt marker is claimed, workspace observers must not run Git status,
repository scans, tests, formatters, IDE indexing, or other workspace operations
until the registered process exits. Monitoring should be limited to the process
handle and `/proc`.

## Terminal disposition

Experiment 006 is a terminal execution failure, not a benchmark result. It has
no automatic terminal report, published artifact, reusable checkpoint, or
scientific metric. The incident record and permanent marker are retained as the
evidence of the one authorized invocation.

The recovery series ends here. Profile `006` will not be retried or altered,
its marker and registration will not be reused, and there will be no Experiment
007 or successor recovery profile. Further portfolio work proceeds outside this
execution lineage.
