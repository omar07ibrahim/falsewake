# Experiment 004: recovery of the registered admission contract

## Status

Experiment 004 was specified and implemented, but it was rejected during
pre-registration verification. It was never registered or run. Its authority
issuer, coordinator, and runner were not invoked; no optimizer update or validation
example occurred; and its permanent attempt marker was never created.

The immutable
[preflight incident](../reports/experiment-004-preflight-incident.json), SHA-256
`d015b749acf6e98e87172372315c78ccfaad48b8ae6b3a29ae426890aa467f50`,
records two internally inconsistent claims in the frozen execution protocol.
Because a frozen contract cannot be silently amended after implementation,
Experiment 004 is closed without a registration. Any recovery requires a new
experiment identifier and a corrected protocol committed before implementation.

## Why this is a new experiment

Experiment 003 was registered and invoked exactly once. Its capability was issued
correctly, but admission failed before the coordinator claimed a parent operation.
The fixed facade exposed `_verified_state` with one positional-only argument:
`_verified_state(registration, /)`. The independently frozen Experiment 003
coordinator required that route to have one positional-or-keyword argument. It
therefore rejected the live function identity before staging preparation, child
creation, training, validation, or publication.

That invocation is terminal. It started no seed child or training process,
performed no optimizer update or validation example, published no artifact or
automatic terminal report, and admitted no reusable checkpoint. The absence of an
automatic report does not make the invocation reusable.

The complete predecessor chain is immutable:

- protocol commit `e47bd581675abe22b529de7bcc825d39e84a00cd`, with
  `configs/experiment-003-execution.json` SHA-256
  `3f48cb48f6c3a56284f272fa9e308ae62b3749df64cb7a581e770da8bdd8218a`;
- implementation commit `4475461d5fd3e5b8969020003424c73bb10d3c9b`;
- registration commit `f7426fb038ae5dc0c24b5141d12d55323ac7c961`, whose
  `configs/experiment-003-run.json` SHA-256 is
  `941a4ec67d2adbe0149861a678fc19e46ac4e2188c126b64c4b9c54f01425b2c`;
  and
- incident commit `96f151d86ae060b402a0ef5d47de7ad8c1191c0a`, containing the
  immutable
  [admission incident](../reports/experiment-003-execution-incident.json) with
  SHA-256
  `0dc24fd211129cd2b81e29fb91ac9e66a0aad25f1deeae332f3b6ea420567688`.

Experiment 004 therefore has a new protocol, profile, registration, transport and
publication namespace, and one-attempt lifecycle. Neither the Experiment 003
registration nor any part of its consumed attempt may be reused.

## Exact scientific inheritance

Experiment 004 retains the Experiment 002 scientific protocol exactly. It does not
retune or replace the data, model, seeds, optimizer, schedule, training population,
augmentation, normalization, validation arithmetic, checkpoint ranking, selected
seed rerun, or gates.

In particular:

- the seeds remain `20260719`, `20260720`, and `20260721`;
- each seed still runs 30 epochs, 313 updates per epoch, and 9,390 updates total;
- the training and validation populations remain 40,027 and 10,583 examples;
- the model remains 53 parameter tensors and 23,724 parameter values;
- all three Experiment 002 scientific configs, the model source, normalization
  artifact, PCM cache, manifests, inventories, and implementation audit retain
  their frozen byte identities; and
- every scientific RNG, history, model-tensor, child-result, and process-guard
  domain retains its `exp002` identity.

The `exp002` domains are scientific inputs, not stale execution labels. Changing
one would alter sampling, augmentation, evidence, or model bytes. Inner training
histories must therefore continue to identify scientific protocol `002`, while the
outer execution and final evidence identify Experiment 004.

All seven Experiment 003 execution modules are also immutable predecessor
references. The intended production delta added one fixed profile `004` to the
shared `experiment_002_run_authority.py` engine without changing profile `002`,
profile `003`, or scientific execution semantics. The frozen protocol also claimed
that removing its enumerated profile symbols and dispatch arms would restore the
incident-commit bytes. Preflight verification proved that enumeration incomplete.

## The admitted interface repair

The failed interface is repaired at the consumer boundary. The Experiment 004
facade deliberately keeps the authority route as one positional-only argument:
`_verified_state(registration, /)`. The Experiment 004 coordinator must accept that
exact live shape: one positional-only argument, zero positional-or-keyword and
keyword-only arguments, no defaults, no keyword defaults, and no variadic
arguments.

This coordinator expectation is the only repair to the mismatched live interface.
The implementation binds profile `004` correctly. The protocol nevertheless
required its facade to become byte-identical to the Experiment 003 facade after
eight case-sensitive substitutions that do not cover the mandatory uppercase
profile bindings. Satisfying that textual recipe would require binding the new
facade to profile `003`, which would be semantically wrong and unsafe.

## Live pre-registration admission proof

The local unit suites that missed the Experiment 003 mismatch are not sufficient
for Experiment 004. Before registration, an isolated-process proof must import the
actual coordinator, authority facade, runner, supervisor, seed worker, final
evidence, and final publication modules. It must inspect the live function objects
and the coordinator's actual route-claim AST as one cross-module contract.

For every one of the 15 parent routes and four wrapper surfaces, the proof checks
the exact `types.FunctionType`, module and function names, positional-only and
positional-or-keyword counts, zero keyword-only arguments, absent defaults and
keyword defaults, and absence of variadic positional or keyword parameters. In
particular, it compares the live Experiment 004 `_verified_state` code object with
the coordinator's one-positional-only expectation. This closes the precise gap
that admitted the incompatible Experiment 003 facade and coordinator separately.

The proof is observational. It must not invoke the registered authority issuer,
the coordinator's claimed closure, the registered coordinator, or the registered
runner. It must neither create nor delete the canonical attempt marker. Passing
this proof is necessary for registration; it is not a registered execution or a
scientific result.

## Preflight outcome

The live 15-route parent matrix and four wrapper routes passed, including the exact
one-positional-only `_verified_state` contract. The P4 targeted suite, Experiment
003 regressions, and shared Experiment 002 authority and execution regressions also
passed. That was not enough to make the frozen machine contract satisfiable.

The independent byte audit found two protocol defects:

1. The facade recipe permits eight literal substitutions and no post-normalization
   edit. After applying all eight, the correct Experiment 004 facade still differs
   from the immutable Experiment 003 reference by 11 added and 11 deleted lines.
   Its required `_EXPERIMENT_004_PROFILE` and
   `_EXPERIMENT_004_SEALED_CHILD_MEMFD_TARGET` bindings cannot be expressed by the
   recipe.
2. The shared-authority recipe lists five modified generic dispatches but omits
   `_frozen_file_bindings`. That sixth dispatch is required because the exact
   Experiment 003 predecessor outcome has no report path or SHA-256. Removing only
   the listed arms therefore cannot restore the claimed baseline bytes.

These are defects in the already committed protocol, not excuses to weaken the
correct implementation. Preflight stopped before the config-only registration,
before the permanent-marker boundary, and before any scientific computation.

## Authority and execution namespaces

Experiment 004 uses fixed authority profile `004`. Callers cannot select an
experiment, path, transport domain, scratch root, issuer, or entrypoint. Capability
state, re-verification, sealed-child issuance, coordinator admission, evidence,
and publication all recheck the retained profile. Capabilities for profiles `002`,
`003`, and `004` are mutually inadmissible at each other's boundaries, and all
profiles share the same process-local issuance latch.

The execution namespace is disjoint from its predecessor:

- runner: `src/falsewake/experiment_004_runner.py`;
- run config: `configs/experiment-004-run.json`;
- scratch: `/home/ubuntu/gitcode/.t/falsewake-experiment-004-scratch`;
- staging: `/home/ubuntu/gitcode/.t/falsewake-experiment-004-staging`;
- report: `reports/experiment-004-training.json`;
- histories: three `experiment-004-seed-…` files and one selected-rerun file; and
- selected model: `models/experiment-004-selected.safetensors`.

The sealed child and activation transports use `FW4CHLD1` and `FW4ACTV1`.
Runtime, source-bundle, and activation-ticket domains use `falsewake-exp004-…`;
sealed origins use `falsewake-sealed://experiment-004/`. The inherited scientific
envelopes retain their `exp002` domains and are accepted only when their current
registration frame and outer profile `004` evidence both verify.

## Permanent one-attempt boundary

Experiment 004 adds a host-persistent parent marker:
`/home/ubuntu/gitcode/.t/falsewake-experiment-004-attempt`. The runner first
classifies the process as the parent or an authenticated sealed child. Only the
parent then claims the marker with `O_CREAT|O_EXCL`, together with `O_WRONLY`,
`O_CLOEXEC`, and `O_NOFOLLOW`, before activating the registered `sys.path` or
importing the run authority or coordinator.

Successful exclusive creation is the attempt-consumption boundary. The parent
writes the fixed marker payload, syncs the file and parent directory, and changes
the marker from mode `0600` to `0444`. An existing marker is rejected without
removal or reuse. Any failure after creation is terminal, and the marker is never
deleted on either success or failure. A sealed child never inspects, creates,
modifies, or deletes it.

Pre-registration checks may verify that the path is absent and safe, but tests must
redirect any creation to a private temporary path. They must never create or
delete the canonical marker. This durable boundary closes the ambiguity exposed by
Experiment 003, where an admission failure consumed the registered invocation
without reaching a publisher capable of writing a terminal report.

## The registration that was not created

The Experiment 004 machine protocol was introduced as the sole change in commit
`c7a8b3c493e05211ff7afa3a7abb974fc1c8b4e2`, a direct child of the Experiment 003
incident. Its committed SHA-256 is
`aba6c1eca84ad33c7751e4768a8ef630ba2c063720fd29938dd1b69abf945a46`.

The intended implementation commit had to descend from that protocol commit,
contain exactly the admitted production delta and its adversarial test surface,
and contain no run config or Experiment 004 managed output. The full portable and
host-bound suites, formatting, lint, type checks, action lint, predecessor hashes,
cross-profile rejections, live route proof, clean tree, and canonical-marker
absence all had to pass before registration. The contradictory parity checks made
that gate impossible, so the registration step was never reached.

Had the gate passed, registration would have been one direct child of the final
implementation commit:

1. its only tree change is `A configs/experiment-004-run.json`;
2. the file is canonical ASCII JSON with non-executable mode `100644`; and
3. it binds the exact implementation, complete source bundle, runtime, invocation,
   frozen machine protocol, scientific inputs, and full Experiment 003 predecessor
   chain.

This offline preflight rejection did not consume a registered invocation, but it
did close the unsound protocol. There is no retry, fallback seed, altered
checkpoint, path override, protocol amendment, or later registration under this
identifier.

## Evidence semantics

The frozen protocol defined the report as the last-linked terminal publication
marker, but no Experiment 004 report or other managed artifact was produced.

- `pass` would have published all four histories, the selected model, and the
  report.
- `gate_failure` would have published the four histories and report, but no
  selected model.
- `execution_failure` would have published only a report with `artifacts: []` and
  `checkpoint_reusable: false`, when enough verified state existed to publish it.

Any outer report would have identified `experiment: "004"` and
`scientific_protocol: "002"`, bound the exact current registration frame, and
carried an exact recursive copy of the immutable Experiment 003 predecessor
record. No such report exists. The preflight incident is infrastructure evidence,
not a scientific result, checkpoint, or authorization to reuse this protocol.
