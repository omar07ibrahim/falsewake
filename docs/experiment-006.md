# Experiment 006: semantic recovery at the pre-registration boundary

## Executive summary

Experiment 006 repairs one execution-admission contract; it does not change the
frozen Experiment 002 model, data, training, or evaluation design. It replaces an
unsatisfiable textual facade-normalization rule with an exact semantic AST plus a
separate truthful-docstring contract. The proof replays all 26 frozen negative
mutations in memory, checks a 13-event/70-call repository-history oracle, and
verifies all 20 directed cross-profile rejection paths. The executed
pre-registration checks passed 343 P6-targeted, 3,012 portable, and 525
host-bound tests, together with strict typing, lint, scoped formatting, and
workflow lint.
No Experiment 006 registration, invocation, attempt marker, model, metric, or
managed output exists at this boundary.

## Status

Experiment 006 is an offline pre-registration candidate. Its execution protocol
was frozen before implementation as the only change in commit
`4adf524f9a3007d5382a851b3419e697d80ceeef`, a direct child of the
terminal Experiment 005 incident. The committed
[`configs/experiment-006-execution.json`](../configs/experiment-006-execution.json)
has SHA-256
`47ce6c6f11c3575944ef8c4ab1edcc0b42fdfa9bf1ede0318bfc5ea61fabf650`,
mode `100644`, and 129,719 canonical ASCII JSON bytes.

This document does not register or launch Experiment 006. There is no
`configs/experiment-006-run.json`, permanent attempt marker, model, training
history, or outcome report. The registered authority issuer, coordinator, and
runner have not been invoked. No optimizer update or validation example has been
observed under this profile.

The production boundary consists of one additive profile in the shared authority
and seven fixed-profile Experiment 006 modules. Their current SHA-256 identities
are:

- shared authority:
  `bcc022601ddb16aef02715f337f200838626996106dc75f349365de28ba40ddb`;
- coordinator:
  `aa4643dcf3bdef64eeadb26fe1dae590472058ccf21d8e01da942cc25095b35d`;
- final evidence:
  `6720741c44e45877c9531360101b1aca0139638f7acf8b81e0747cce77462d59`;
- final publication:
  `1128242c9ea1eb31f9802f0d45cc672bd31d1d5f0135b4310692255bf0899417`;
- authority facade:
  `fe1be07108c5f35e82d6ace289c09a2a25ee5c54472f49a9fb7ec758177160e2`;
- runner:
  `a2fdb9f13aa909d0c8de5f457c6af0ea26b391c41aefc28b6995ac259075fcb2`;
- seed worker:
  `25dfe1659dcda36dd73bd5f8e5ab6f6f663572e4abf66954eb73e1d075f5b86b`;
  and
- supervisor:
  `615c3d1c4bd3ac96f61b5cf12e2e659881df51255157afcf48425d692353cf65`.

## Why this is a new experiment

Experiment 005 ended before registration. Its frozen facade-normalization recipe
could not express the truthful profile-005 `_verified_state` docstring. The
candidate correctly said:

```python
"""Return shared retained state only when its exact profile is ``005``."""
```

The protocol only replaced the substring `profile ``005```; the intervening
`is ` made that replacement inapplicable. Leaving the stale profile `004` would
have satisfied byte parity by making the documentation false. Rewriting a frozen
contract after seeing its implementation would instead erase the failed
prediction.

The immutable
[`reports/experiment-005-preflight-incident.json`](../reports/experiment-005-preflight-incident.json),
SHA-256
`3de6fbbd6912bee47f3ef01f5371b57ae8d5942b6fc9915fc0ec243b9e931e3d`,
records that rejection in commit
`594e2c491c057394f18860c8362a51190460645f`. Experiment 005 therefore has
no registration, consumed attempt, checkpoint, or reusable output. Experiment
006 preserves that terminal predecessor byte-for-byte and uses a new authority
profile, execution namespace, future registration contract and path, transport
domain, and permanent one-attempt boundary.

## Exact scientific inheritance

Experiment 006 does not retune after the Experiment 005 protocol preflight
rejection. It retains the Experiment 002 scientific protocol exactly: the same
data split, three training seeds, augmentation, normalization,
23,724-parameter causal model, optimizer, schedule, ranking rule, selected-seed
rerun, and gates.

In particular:

- seeds remain `20260719`, `20260720`, and `20260721`;
- each seed has 30 epochs, 313 updates per epoch, and 9,390 updates in total;
- each epoch has 40,027 training examples and validation has 10,583 examples;
- the model has 53 parameter tensors and 23,724 parameter values; and
- Python 3.12.3, NumPy 2.5.1, PyTorch 2.13.0+cpu, and safetensors 0.8.0
  remain fixed.

The Speech Commands archive, exact 105,829-record manifest, and 2,995,776,948-byte
PCM cache remain external inputs rather than repository artifacts. Their required
SHA-256 values are respectively
`af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58`,
`d28e6993101bd6bc452033bcb7355b25e3b84ab5c51a1458cc097dc93c60f78b`,
and
`b56270a4b99d235c62c162545b139a79b7cf3dc157259a3bbf5678d8a451f653`.
The `exp002` inner domains are retained because they determine scientific
sampling, augmentation, histories, and model bytes. Outer authority, transport,
and publication evidence is profile `006`.

## Replacing textual imitation with a semantic contract

The failed Experiment 005 recipe treated a small authority facade as text to be
normalized into its predecessor. Experiment 006 specifies the interface it
actually needs.

The frozen `semantic_ast_v1` projection parses Python 3.12 syntax, excludes source
locations and only the leading module and function docstring nodes, and preserves
statement order. It then requires exact imports, aliases, exports, eight
synchronous non-generator routes, signatures, direct-call bodies, and a
closed top-level surface. Defaults, wrappers, async functions, generators,
additional statements, module-level dynamic `__getattr__`, and exception
wrappers are forbidden.

Docstrings are verified separately rather than discarded as irrelevant prose.
There are nine exact scopes: eight profile-bearing scopes name `006` truthfully,
while `_required_child_bundle_seals` has exact profile-neutral wording. Stale
claims for profiles `002` through `005` are rejected. This separation is
deliberate: changing a truthful docstring to `005` leaves the semantic AST
projection unchanged but must still fail admission.

Before the protocol was frozen, a synthetic witness satisfied ten positive
checks. Twenty-six one-at-a-time negative mutations then demonstrated that the
validator rejects signature drift, wrong-profile delegation, extra wrapper
logic, async or generator conversion, stale documentation, omitted dispatches,
unexpected Git operations, swallowed history faults, self-reference, and early
registration topology. The operators and expected outcomes are part of the
frozen protocol rather than tests invented after implementation.

## Shared authority: positive semantics and inverse restoration

The shared authority change is bounded in two directions. The positive contract
requires exactly 17 added profile-006 symbols: ten constant assignments, six
simple fixed-profile functions, and one implementation-history verifier. It
allows insertions in exactly six existing generic dispatches:

1. `_AUTHORITY_PROFILES`;
2. `_expected_frozen_bindings`;
3. `_frozen_file_bindings`;
4. `_require_authority_profile`;
5. `_require_invocation_registration`; and
6. `_verify_committed_repository`.

No seventh dispatch or arbitrary logic inside an allowed dispatch is admitted.
The inverse contract removes those enumerated additions and must restore the
terminal Experiment 005 shared-authority blob exactly, both as bytes and as a
semantic AST. This closes the omission that made the Experiment 004 restoration
claim unsatisfiable while also proving that existing profiles retain their
predecessor behavior.

The implementation-history verifier is not accepted merely because its source
looks plausible. Its AST call closure maps 13 ordered logical history events to
exactly 70 intercepted primitive calls through `_git`, `_git_process`,
`_committed_blob`, and the recursive Experiment 005 history verifier. An isolated
fake-Git oracle verifies the exact event order, arguments, results, P5 recursive
bindings, source delta, file modes, ancestry, and absence of transient P4/P5/P6
managed outputs anywhere between the protocol and implementation commits. An
injected result fault for each required logical event must reject; extra,
missing, duplicated, swapped, swallowed, or unmapped calls also reject. The
history verifier is observational and neither reads nor mutates any attempt
marker.

## Live execution boundary

The pre-registration proof imports the actual seven Experiment 006 modules in an
isolated process without calling the issuer, coordinator, or runner. It verifies
15 parent routes and four wrapper routes as one cross-module contract, including
exact function identity, module, name, positional-only and
positional-or-keyword counts, absent defaults, absent keyword defaults, no
variadic parameters, and synchronous non-generator code objects.

All five authority profiles share one capability type, one process-local registry,
and one issuance latch. Every ordered pair of distinct profiles is rejected:
five profiles produce exactly 20 directed cross-profile rejection cases.
Capabilities cannot be replayed from `002`, `003`, `004`, or `005` into the
Experiment 006 facade, child transport, coordinator, evidence builder, or
publication boundary.

The final-evidence and publication tests bind the exact terminal P5 predecessor
object, not a simplified status label. They also adversarially exercise evidence
schemas, canonical serialization, immutable fields, descriptor-based publication,
partial I/O, symlink and hard-link attacks, destination races, post-write
verification, cleanup durability, and profile-specific output ownership using
private temporary roots.

## Namespace and permanent attempt boundary

Experiment 006 owns disjoint paths for its future registration, scratch and
staging roots, three seed histories, selected-seed rerun history, selected model,
and terminal report. It also has profile-specific authority, source-bundle,
activation, runtime, sealed-child transport, and outer evidence identities.
History, envelope, and model-tensor digest domains remain the frozen `exp002`
scientific identities.

The parent bootstrap must claim
`/home/ubuntu/gitcode/.t/falsewake-experiment-006-attempt` with
`O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC|O_WRONLY` before activating the registered
source tree or importing the authority and coordinator. It writes a fixed payload,
syncs the file and parent directory, and changes the mode from `0600` to `0444`.
Successful exclusive creation consumes the sole attempt. An existing marker is
never removed or reused; any later failure is terminal. Authenticated children
never inspect, create, modify, or delete the marker.

Tests redirect marker creation to private temporary paths. Pre-registration
verification may only observe that the canonical path is absent; it must never
create or delete it.

## Pre-registration verification

Before a registration commit can exist, the final clean admission commit must
contain the complete production delta, protocol proof, adversarial execution
tests, and this document while containing no run config or managed output. The
gate requires:

- the frozen protocol topology, mode, bytes, and terminal P5 predecessor chain;
- exact Experiment 002 scientific inputs and all predecessor source identities;
- the semantic facade and truthful docstring manifests;
- positive shared-authority semantics and exact inverse restoration;
- the 13-event/70-primitive fake-Git trace and transient-path scan;
- the 15-plus-4 live route matrix and all 20 cross-profile rejections;
- portable and host-bound test suites with warnings treated as errors;
- strict mypy, Ruff, Ruff formatting, and actionlint;
- a clean Git tree, absent P4/P5/P6 managed outputs, and absent P4/P5/P6
  canonical markers.

The completed pre-registration checks are:

- protocol proof commit:
  `fe21f22feb4d258f291c4d8c8bfc9646f2618227`;
- [`tests/test_experiment_006_protocol.py`](../tests/test_experiment_006_protocol.py)
  SHA-256:
  `7f4510ff9786db856a5222a7455a75db9a23c603fa39aecf811c65a942cfd2f4`;
- P6 targeted suite: 343 passed with warnings treated as errors;
- portable suite: 3,012 passed with warnings treated as errors;
- host-bound suite: 525 passed with warnings treated as errors;
- strict mypy: no issues in 157 source files;
- Ruff lint: passed for the repository;
- Ruff formatting: passed for all 19 Python files added or modified since the
  frozen implementation base commit
  `594e2c491c057394f18860c8362a51190460645f`; and
- actionlint 1.7.12: passed for `.github/workflows/ci.yml` after verifying the
  official Linux amd64 release checksum
  `8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8`.

The protocol explicitly qualifies the two test suites as full, but records
`ruff_format_required` without a repository-wide command or scope. Formatting is
therefore checked over the complete admitted Python delta from its exact frozen
base. Eleven legacy files outside that delta do not satisfy the current Ruff
formatter, including four source files whose modification would violate the
simultaneously frozen `other_src_falsewake_changes: forbidden` boundary.
Repository-wide formatting is not claimed.

The admission commit is defined symbolically as the final clean commit containing
this document and the complete admitted delta. Its own SHA cannot appear here
without creating a forbidden self-reference; the direct-child registration
config, if admitted, binds it through the registration commit's parent.

## Registration that does not yet exist

Only a successful final admission permits one direct-child registration commit.
Its sole change must be
`A configs/experiment-006-run.json`, canonical ASCII JSON at mode `100644`.
That file must bind the exact admission commit, source bundle, runtime, invocation
contract, execution protocol, scientific inputs, and complete predecessor chain.

The registration commit—not this execution protocol or document—would authorize
one registered invocation. A preflight rejection would require a new experiment
identifier. Once the permanent marker is created, success or failure is terminal:
there is no retry, fallback seed, checkpoint reuse, altered config, or second
launch under profile `006`.
