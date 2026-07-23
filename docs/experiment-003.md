# Experiment 003: a clean execution after an infrastructure failure

## Status

Experiment 003 is specified but not registered or run. The existence of this
document and
[`configs/experiment-003-execution.json`](../configs/experiment-003-execution.json)
does not authorize training. The one permitted execution starts only after a
separate implementation commit passes the full test gate and a direct child commit
adds only `configs/experiment-003-run.json`.

## Why this is a new experiment

Experiment 002 was launched exactly once. It ended with the immutable
`registered_execution/seed_selection_failed` report and admitted no scientific
result, history, model, or reusable checkpoint. The failure was caused by a
supervisor assumption about Linux `/proc`: an exiting userspace thread-group leader
can release its memory map before its task state becomes `Z`, temporarily leaving
the leader's `status` without `VmRSS`.

The failure report, registration, source bundle, and
[incident record](../reports/experiment-002-execution-incident.json) remain
unchanged. Running the Experiment 002 entrypoint again is forbidden. Experiment 003
therefore has a new registration, authority profile, runner, transport domains,
scratch and staging roots, publication paths, and one-attempt lifecycle.

This is not a scientific retune. It is a new execution identity for the unchanged
Experiment 002 scientific protocol.

## Exact scientific inheritance

The data, model, seeds, optimizer, schedule, training population, augmentation,
normalization, validation arithmetic, checkpoint ranking, winner rerun, and gates
remain those of Experiment 002. In particular:

- the seeds remain `20260719`, `20260720`, and `20260721`;
- every seed still runs 30 epochs, 313 updates per epoch, and 9,390 updates total;
- the model still has 53 parameter tensors and 23,724 parameter values;
- all Speech Commands identities, the PCM cache, and the normalization artifact
  retain their existing byte counts and SHA-256 values; and
- all scientific RNG and digest domains retain their `exp002` bytes.

The last point is intentional. Replacing an `exp002` HMAC or sampling domain with
`exp003` would change the training examples or augmentation decisions and would be
a scientific change. Experiment 003 imports the inherited scientific modules under
their existing names and treats their inner histories as
`scientific_protocol: "002"`. Only the outer execution envelope and final report
identify Experiment 003.

The machine protocol binds the three scientific configs, model source,
normalization artifact and report, PCM-cache report, implementation audit, external
input identities, runtime versions, and the predecessor source bundle. Every
predecessor source path other than the authority and supervisor must remain
byte-identical to implementation commit
`21bbcc56e898beb44edc8e3ff775fa9642f87577`.

- `experiment_002_run_authority.py` may be extended in place only as a
  profile-bound shared authority engine.
- `experiment_002_supervisor.py` must equal the repaired bytes in commit
  `650eefbf9f82b207ed58e3b1a1eac41197466b41`, rather than its failing bytes in
  the earlier implementation commit.

The Experiment 002 supervisor remains the repaired reference. A new
`experiment_003_supervisor.py` must differ from it only where the fixed runner,
scratch, staging, and namespace identities require it. The other six allowed
production additions are enumerated exactly in the machine protocol; arbitrary
files under the prefix are not admitted. Any change to a scientific input,
hyperparameter, gate, RNG domain, data route, or model requires a different
experiment identifier and a new protocol committed before observing its metrics.

## The permitted infrastructure delta

The repaired supervisor samples every accessible userspace task when the
thread-group leader does not expose `VmRSS`, uses the maximum live task RSS, and
bounds a total RSS-unavailable interval to 100 ms. CPU affinity, wall time, output
bytes, descendant discovery, and process-cycle checks continue throughout that
interval. A later valid sample resumes live RSS enforcement, and terminal
`wait4.ru_maxrss` remains authoritative.

The repair commit is
`650eefbf9f82b207ed58e3b1a1eac41197466b41`; its supervisor SHA-256 is
`f441f2ead5566094364572138d69256a072165ffb2cc80a71340b4087ba5a8f0`.
The failing implementation's supervisor SHA-256 was
`417df7117947b41f2857c3400c6fa176a2749a701357f2bb184a3928bbd2c414`.
Those two identities and the exact failure evidence are part of the Experiment 003
admission contract.

## Authority and namespace boundaries

The Experiment 003 authority is a fixed-profile facade over a shared opaque
registration capability. No caller can provide an experiment ID, path, domain, or
profile. The capability state itself records profile `003`; sealed source-bundle
headers, virtual origins, runtime fingerprints, re-verification, child activation,
the coordinator, and publication all recheck that profile. The inherited
child-result payload binds the complete current registration frame; the outer
Experiment 003 evidence layer separately rechecks the retained profile before
accepting it.

The execution namespace is disjoint from Experiment 002:

- runner: `src/falsewake/experiment_003_runner.py`;
- run config: `configs/experiment-003-run.json`;
- scratch: `/home/ubuntu/gitcode/.t/falsewake-experiment-003-scratch`;
- staging: `/home/ubuntu/gitcode/.t/falsewake-experiment-003-staging`;
- report: `reports/experiment-003-training.json`;
- histories: three `experiment-003-seed-…` files and one selected-rerun file; and
- selected model: `models/experiment-003-selected.safetensors`.

The source-bundle and activation transport uses `FW3…` magic values and
`falsewake-exp003-…` domains. The byte-identical process guard and child-result
payload remain explicitly marked as inherited Experiment 002 internals; their
registration frame binds them to the new run before the outer Experiment 003
evidence accepts them. The old Experiment 002 run registration, source bundles,
activation tickets, temporary names, and managed outputs are invalid at
Experiment 003 execution boundaries. The Experiment 002 training, numerics, and
trainer configs are instead mandatory inherited scientific inputs. Experiment 003
cleanup and publication never remove, rename, replace, or change the mode of an
Experiment 002 path.

## Registration and one-attempt rule

The implementation commit must descend from the repair commit, contain the complete
authority, runner, supervisor, worker, evidence, publication, and adversarial test
surface, and contain no Experiment 003 run config or managed result.

After the full portable and host-bound gates pass, registration is one direct child
commit:

1. it has the implementation commit as its only parent;
2. its only tree change is `A configs/experiment-003-run.json`;
3. that file is a non-executable `100644` canonical ASCII JSON blob; and
4. it binds the exact implementation commit, complete source bundle, runtime,
   invocation, external inputs, this execution protocol, predecessor evidence, and
   inherited scientific bytes.

The registered runner has no retry, seed, path, callback, resource, or
hyperparameter override. An offline preflight rejection occurs before the
registered invocation, publishes no terminal evidence, and must be resolved before
launch. Once the exact registered runner is invoked, the parent attempt latch is
consumed before route claim and a fresh staging namespace supplies cross-process
ownership. Admission failure after invocation, child failure, gate failure,
publication failure, and success all exhaust Experiment 003. A failure before the
publisher has enough verified state may leave no automatic terminal report; that
does not authorize another invocation. A second run under this identifier is never
allowed.

## Evidence semantics

The report is linked last and is the only terminal commit marker.

- `pass` publishes the three seed histories, the selected-seed rerun history, the
  selected model, and the report.
- `gate_failure` publishes the four histories and report, but no selected model.
- `execution_failure` publishes only a report with `artifacts: []` and
  `checkpoint_reusable: false`.

Every outer report identifies `experiment: "003"`,
`scientific_protocol: "002"`, the current registration frame, and the immutable
Experiment 002 predecessor frame. Inner training histories retain their honest
Experiment 002 scientific identity. A raw predecessor artifact cannot be relabeled
or republished: the inherited payload must bind the current registration frame, and
the Experiment 003 activation and outer evidence profile must both verify.

No terminal status permits a fallback seed, epoch, checkpoint, hyperparameter, or
second publication attempt.
