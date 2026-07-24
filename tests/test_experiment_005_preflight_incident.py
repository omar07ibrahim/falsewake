from __future__ import annotations

import ast
import difflib
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "experiment-005-preflight-incident.json"
REPORT_COMMIT = "594e2c491c057394f18860c8362a51190460645f"
REPORT_PARENT = "9ae549ccd477b13b9531f7c44d854d34549fd9c3"
REPORT_SHA256 = "3de6fbbd6912bee47f3ef01f5371b57ae8d5942b6fc9915fc0ec243b9e931e3d"
REPORT_GIT_BLOB = "4c83a4d2233c18c5b69f11aa9302c41572d16cea"
REPORT_BYTES = 6_612

PROTOCOL_PATH = ROOT / "configs" / "experiment-005-execution.json"
PROTOCOL_COMMIT = "0b6bf2cac2d4f6a04d6ced596bf8f835660eb072"
PROTOCOL_PARENT = "462aeba306a0612fd6d64884e323d72db3569a89"
PROTOCOL_SHA256 = "824d1677cf8f75567cf1915381f0a9bbc778a74cba879ace80733c3ceb48e54d"
PROTOCOL_GIT_BLOB = "38221d0314fd0b2cb0946c8ea24f6e74956a0c7a"

IMPLEMENTATION_COMMIT = "04b634451d129faadadd921215123547e3467d65"
IMPLEMENTATION_PARENT = "86210da504ba9842f34546fb471e2d6dc0b9b0ba"
PROOF_COMMIT = "1e20d6cf210a2d4a59cfc737a14eaccfc94a5b74"
PROOF_PATH = ROOT / "tests" / "test_experiment_005_protocol.py"
PROOF_SHA256 = "cf6d4ee2fcba31de37774b591571abb2d86998cc0f2a3e99a40abe1af4af5414"
PROOF_GIT_BLOB = "893ae5b7544f0cb142436adf20122221c6cc4859"
BOUNDARY_COMMIT = REPORT_PARENT
BOUNDARY_PATH = ROOT / "docs" / "experiment-005.md"
BOUNDARY_SHA256 = "c9852a028f572e33b2379b98786f5c72e41ddae1300f03742ad977947a1dd600"
BOUNDARY_GIT_BLOB = "e79d6ed83c8ce3f6230086b8ab273d1546a4a248"

P4_REFERENCE_PATH = ROOT / "src" / "falsewake" / "experiment_004_run_authority.py"
P4_REFERENCE_SHA256 = "cacab13729f14bfdce10afa5b53b2a09116cf56abe21aee12ac9220a6ad2e5b5"
FACADE_PATH = ROOT / "src" / "falsewake" / "experiment_005_run_authority.py"
FACADE_SHA256 = "87d18e901b4b814ce31c7c17b60b6aa54f837edc84ad3bb5ee1af386df006f5e"
NORMALIZED_FACADE_SHA256 = (
    "9778140580667e95e120a4a8472d8b79ffb7034c3886c208fc3fd3c4da966c86"
)
SHARED_AUTHORITY_SHA256 = (
    "937ad120dd5828cb4029cc1382df2a0a7c06aef14c1d37bfdb848d93993416af"
)

P4_MARKER = Path("/home/ubuntu/gitcode/.t/falsewake-experiment-004-attempt")
P5_MARKER = Path("/home/ubuntu/gitcode/.t/falsewake-experiment-005-attempt")

PRODUCTION_BINDINGS: tuple[dict[str, str], ...] = (
    {
        "path": "src/falsewake/experiment_005_coordinator.py",
        "sha256": "4ad78de14ce12acdb63b4e103dd920ce878d7e60a06fa1c5893027d17401f812",
        "blob": "90aac5457a7f0a09010ee5a504df7399a5ffb680",
    },
    {
        "path": "src/falsewake/experiment_005_final_evidence.py",
        "sha256": "1c99757fc9b78e612891a9d49e7dc40c2d0b90599a40927e4cd41dff781f355d",
        "blob": "44580948c078118c232cbafc3caed82c020f3392",
    },
    {
        "path": "src/falsewake/experiment_005_final_publication.py",
        "sha256": "84f7aba57f8ec0c096e5f18a03a8118800f53a9cfe4910c554bea677a9f79f07",
        "blob": "90e26b9918d89f1ecfdd04dbe523d73aa9fe8589",
    },
    {
        "path": "src/falsewake/experiment_005_run_authority.py",
        "sha256": FACADE_SHA256,
        "blob": "2b74d13a9e210a52767cb10968e35c5659a67ed4",
    },
    {
        "path": "src/falsewake/experiment_005_runner.py",
        "sha256": "eeafface36baaabba5e1479750674884a069b41ccaeeb9dee9ccbc4b179a5336",
        "blob": "d286927304ba58069b9a84ceae05e5054f74ba48",
    },
    {
        "path": "src/falsewake/experiment_005_seed_worker.py",
        "sha256": "0840a62f0f4327eec7649d17ad334edfa6941d6146bff6fe50e2d693f0b80f96",
        "blob": "7c302b503ec8e67ea2b56bf53ce76b5307c42112",
    },
    {
        "path": "src/falsewake/experiment_005_supervisor.py",
        "sha256": "7c8d58566c2b5e904ebf7f1e5fa1ab3b1c79c4387b38f296b08ef5ce7e312e71",
        "blob": "560b24c93c2e203416dc9fb2f4151cf22020c412",
    },
)
SHARED_AUTHORITY_BINDING = {
    "path": "src/falsewake/experiment_002_run_authority.py",
    "sha256": SHARED_AUTHORITY_SHA256,
    "blob": "94fb83bb4a2deeb88ede35d0096caa22d55a5002",
}

MANAGED_OUTPUTS = (
    "configs/experiment-005-run.json",
    "models/experiment-005-selected.safetensors",
    "reports/experiment-005-seed-20260719-history.json",
    "reports/experiment-005-seed-20260720-history.json",
    "reports/experiment-005-seed-20260721-history.json",
    "reports/experiment-005-selected-rerun-history.json",
    "reports/experiment-005-training.json",
)

EXPECTED_SOURCE_DELTA = b"".join(
    status + b"\0" + path.encode("ascii") + b"\0"
    for status, path in (
        (b"M", "src/falsewake/experiment_002_run_authority.py"),
        *((b"A", binding["path"]) for binding in PRODUCTION_BINDINGS),
    )
)

EXPECTED_RECOVERY_REQUIREMENTS = (
    (
        "Preserve the Experiment 005 protocol, implementation, protocol proof, "
        "preflight boundary, and this incident as immutable evidence."
    ),
    (
        "Do not create an Experiment 005 registration or invoke its authority "
        "issuer, coordinator, runner, permanent attempt marker, or managed outputs."
    ),
    (
        "Use only a new Experiment 006 identifier and disjoint execution namespace "
        "for any recovery."
    ),
    (
        "Prefer an exact semantic AST facade contract; if byte normalization is "
        "retained, add an explicit profile is ``006`` to profile is ``005`` "
        "substitution."
    ),
    (
        "Freeze the corrected Experiment 006 protocol and pass its complete "
        "pre-registration gate before creating any registration."
    ),
)


def _reject_duplicate_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_float(value: str) -> None:
    raise ValueError(f"JSON floats are forbidden: {value}")


def _strict_json(payload: bytes) -> dict[str, Any]:
    assert not payload.startswith(b"\xef\xbb\xbf")
    document = json.loads(
        payload.decode("ascii"),
        object_pairs_hook=_reject_duplicate_key,
        parse_float=_reject_float,
        parse_constant=_reject_float,
    )
    assert type(document) is dict
    return document


def _canonical(document: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(*arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=30,
    )
    return completed.stdout


def _report() -> dict[str, Any]:
    return _strict_json(REPORT_PATH.read_bytes())


def _without_docstring(function: ast.FunctionDef) -> list[ast.stmt]:
    statements = list(function.body)
    if (
        statements
        and isinstance(statements[0], ast.Expr)
        and isinstance(statements[0].value, ast.Constant)
        and type(statements[0].value.value) is str
    ):
        return statements[1:]
    return statements


def test_report_is_strict_duplicate_free_canonical_ascii_with_exact_schema() -> None:
    payload = REPORT_PATH.read_bytes()
    assert len(payload) == REPORT_BYTES
    assert _sha256(payload) == REPORT_SHA256
    report = _strict_json(payload)
    assert _canonical(report) == payload
    assert tuple(report) == (
        "attempt",
        "experiment",
        "implementation",
        "incident",
        "kind",
        "managed_outputs_absent",
        "preflight_boundary",
        "preflight_evidence",
        "protocol",
        "recovery_requirements",
        "schema_version",
        "status",
        "violations",
    )
    assert report["schema_version"] == 1
    assert report["experiment"] == "005"
    assert report["incident"] == "pre_registration_protocol_rejection"
    assert report["kind"] == "execution_protocol_preflight_incident"
    assert report["status"] == "preflight_rejected"
    assert len(report["violations"]) == 1
    assert report["violations"][0]["code"] == (
        "facade_normalization_recipe_cannot_express_truthful_profile_005"
    )


def test_report_only_commit_parent_mode_blob_size_and_payload_are_exact() -> None:
    payload = REPORT_PATH.read_bytes()
    assert (
        _git(
            "rev-list",
            "--parents",
            "-n",
            "1",
            REPORT_COMMIT,
        )
        == f"{REPORT_COMMIT} {REPORT_PARENT}\n".encode()
    )
    assert (
        _git(
            "diff-tree",
            "--no-ext-diff",
            "--no-textconv",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-z",
            REPORT_PARENT,
            REPORT_COMMIT,
        )
        == b"A\0reports/experiment-005-preflight-incident.json\0"
    )
    assert (
        _git(
            "ls-tree",
            REPORT_COMMIT,
            "--",
            "reports/experiment-005-preflight-incident.json",
        )
        == (
            f"100644 blob {REPORT_GIT_BLOB}\t"
            "reports/experiment-005-preflight-incident.json\n"
        ).encode()
    )
    assert (
        _git(
            "cat-file",
            "-s",
            f"{REPORT_COMMIT}:reports/experiment-005-preflight-incident.json",
        )
        == f"{REPORT_BYTES}\n".encode()
    )
    assert (
        _git(
            "show",
            f"{REPORT_COMMIT}:reports/experiment-005-preflight-incident.json",
        )
        == payload
    )


def test_protocol_implementation_proof_and_boundary_chain_is_exact() -> None:
    report = _report()
    assert (
        _git(
            "rev-list",
            "--parents",
            "-n",
            "1",
            PROTOCOL_COMMIT,
        )
        == f"{PROTOCOL_COMMIT} {PROTOCOL_PARENT}\n".encode()
    )
    assert (
        _git(
            "diff-tree",
            "--no-ext-diff",
            "--no-textconv",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-z",
            PROTOCOL_PARENT,
            PROTOCOL_COMMIT,
        )
        == b"A\0configs/experiment-005-execution.json\0"
    )
    assert (
        _git(
            "ls-tree",
            PROTOCOL_COMMIT,
            "--",
            "configs/experiment-005-execution.json",
        )
        == (
            f"100644 blob {PROTOCOL_GIT_BLOB}\tconfigs/experiment-005-execution.json\n"
        ).encode()
    )
    protocol_payload = PROTOCOL_PATH.read_bytes()
    assert _sha256(protocol_payload) == PROTOCOL_SHA256
    assert (
        _git(
            "show",
            f"{PROTOCOL_COMMIT}:configs/experiment-005-execution.json",
        )
        == protocol_payload
    )
    assert report["protocol"] == {
        "commit": PROTOCOL_COMMIT,
        "mode": "100644",
        "path": "configs/experiment-005-execution.json",
        "sha256": PROTOCOL_SHA256,
    }

    assert (
        _git(
            "rev-list",
            "--parents",
            "-n",
            "1",
            IMPLEMENTATION_COMMIT,
        )
        == f"{IMPLEMENTATION_COMMIT} {IMPLEMENTATION_PARENT}\n".encode()
    )
    assert (
        subprocess.run(
            (
                "git",
                "merge-base",
                "--is-ancestor",
                PROTOCOL_COMMIT,
                IMPLEMENTATION_COMMIT,
            ),
            cwd=ROOT,
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )

    assert (
        _git(
            "rev-list",
            "--parents",
            "-n",
            "1",
            PROOF_COMMIT,
        )
        == f"{PROOF_COMMIT} {IMPLEMENTATION_COMMIT}\n".encode()
    )
    assert (
        _git(
            "diff-tree",
            "--no-ext-diff",
            "--no-textconv",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-z",
            IMPLEMENTATION_COMMIT,
            PROOF_COMMIT,
        )
        == b"A\0tests/test_experiment_005_protocol.py\0"
    )
    assert (
        _git(
            "ls-tree",
            PROOF_COMMIT,
            "--",
            "tests/test_experiment_005_protocol.py",
        )
        == (
            f"100644 blob {PROOF_GIT_BLOB}\ttests/test_experiment_005_protocol.py\n"
        ).encode()
    )
    proof_payload = _git(
        "show",
        f"{PROOF_COMMIT}:tests/test_experiment_005_protocol.py",
    )
    assert _sha256(proof_payload) == PROOF_SHA256
    assert report["implementation"]["protocol_proof"] == {
        "commit": PROOF_COMMIT,
        "path": "tests/test_experiment_005_protocol.py",
        "sha256": PROOF_SHA256,
    }

    assert (
        _git(
            "rev-list",
            "--parents",
            "-n",
            "1",
            BOUNDARY_COMMIT,
        )
        == f"{BOUNDARY_COMMIT} {PROOF_COMMIT}\n".encode()
    )
    assert (
        _git(
            "diff-tree",
            "--no-ext-diff",
            "--no-textconv",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-z",
            PROOF_COMMIT,
            BOUNDARY_COMMIT,
        )
        == b"A\0docs/experiment-005.md\0"
    )
    assert (
        _git(
            "ls-tree",
            BOUNDARY_COMMIT,
            "--",
            "docs/experiment-005.md",
        )
        == (f"100644 blob {BOUNDARY_GIT_BLOB}\tdocs/experiment-005.md\n").encode()
    )
    boundary_payload = BOUNDARY_PATH.read_bytes()
    assert _sha256(boundary_payload) == BOUNDARY_SHA256
    assert (
        _git(
            "show",
            f"{BOUNDARY_COMMIT}:docs/experiment-005.md",
        )
        == boundary_payload
    )
    assert report["preflight_boundary"] == {
        "commit": BOUNDARY_COMMIT,
        "mode": "100644",
        "path": "docs/experiment-005.md",
        "sha256": BOUNDARY_SHA256,
    }


def test_implementation_has_exact_shared_modification_and_seven_source_additions() -> (
    None
):
    report = _report()
    assert report["implementation"]["commit"] == IMPLEMENTATION_COMMIT
    assert (
        _git(
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--name-status",
            "-z",
            PROTOCOL_PARENT,
            IMPLEMENTATION_COMMIT,
            "--",
            "src/falsewake",
        )
        == EXPECTED_SOURCE_DELTA
    )
    assert report["implementation"]["production_sources"] == [
        {
            "path": binding["path"],
            "sha256": binding["sha256"],
        }
        for binding in PRODUCTION_BINDINGS
    ]

    for binding in PRODUCTION_BINDINGS:
        path = binding["path"]
        committed = _git("show", f"{IMPLEMENTATION_COMMIT}:{path}")
        current = (ROOT / path).read_bytes()
        assert committed == current
        assert _sha256(committed) == binding["sha256"]
        assert (
            _git(
                "ls-tree",
                IMPLEMENTATION_COMMIT,
                "--",
                path,
            )
            == f"100644 blob {binding['blob']}\t{path}\n".encode()
        )

    shared_path = SHARED_AUTHORITY_BINDING["path"]
    shared = _git("show", f"{IMPLEMENTATION_COMMIT}:{shared_path}")
    assert _sha256(shared) == SHARED_AUTHORITY_BINDING["sha256"]
    assert (
        _git(
            "ls-tree",
            IMPLEMENTATION_COMMIT,
            "--",
            shared_path,
        )
        == (f"100644 blob {SHARED_AUTHORITY_BINDING['blob']}\t{shared_path}\n").encode()
    )
    assert report["implementation"]["shared_authority"] == {
        "path": shared_path,
        "sha256": SHARED_AUTHORITY_SHA256,
    }


def test_attempt_counters_outputs_and_p4_p5_markers_are_exactly_absent() -> None:
    report = _report()
    assert report["attempt"] == {
        "canonical_marker": str(P5_MARKER),
        "canonical_marker_present": False,
        "optimizer_updates": 0,
        "registered_attempt_consumed": False,
        "registered_authority_issuer_invoked": False,
        "registered_coordinator_invoked": False,
        "registered_invocation_count": 0,
        "registered_runner_invoked": False,
        "validation_examples": 0,
    }
    assert tuple(report["managed_outputs_absent"]) == MANAGED_OUTPUTS
    for path in MANAGED_OUTPUTS:
        assert not _git("ls-tree", REPORT_COMMIT, "--", path)
        assert not os.path.lexists(ROOT / path)
    assert not os.path.lexists(P4_MARKER)
    assert not os.path.lexists(P5_MARKER)


def test_frozen_normalization_has_the_exact_single_truthful_line_defect() -> None:
    report = _report()
    protocol = _strict_json(PROTOCOL_PATH.read_bytes())
    candidate = _git(
        "show",
        f"{IMPLEMENTATION_COMMIT}:src/falsewake/experiment_005_run_authority.py",
    ).decode()
    assert candidate == FACADE_PATH.read_text(encoding="utf-8")
    assert _sha256(candidate.encode()) == FACADE_SHA256
    substitutions = protocol["execution_delta"]["parity_contract"]["normalization"][
        "ordered_literal_substitutions"
    ]
    assert len(substitutions) == 12
    for substitution in substitutions:
        candidate = candidate.replace(
            substitution["candidate"],
            substitution["reference"],
        )
    reference = P4_REFERENCE_PATH.read_text(encoding="utf-8")
    assert _sha256(reference.encode()) == P4_REFERENCE_SHA256
    assert _sha256(candidate.encode()) == NORMALIZED_FACADE_SHA256

    reference_line = (
        '    """Return shared retained state only when its exact profile '
        'is ``004``."""\n'
    )
    candidate_line = (
        '    """Return shared retained state only when its exact profile '
        'is ``005``."""\n'
    )
    opcodes = difflib.SequenceMatcher(
        None,
        reference.splitlines(keepends=True),
        candidate.splitlines(keepends=True),
        autojunk=False,
    ).get_opcodes()
    changed = [opcode for opcode in opcodes if opcode[0] != "equal"]
    assert len(changed) == 1
    tag, reference_first, reference_last, candidate_first, candidate_last = changed[0]
    assert tag == "replace"
    assert reference.splitlines(keepends=True)[reference_first:reference_last] == [
        reference_line
    ]
    assert candidate.splitlines(keepends=True)[candidate_first:candidate_last] == [
        candidate_line
    ]
    assert candidate.replace(candidate_line, reference_line) == reference

    normalization = report["preflight_evidence"]["facade_normalization"]
    assert normalization == {
        "candidate_normalized_sha256": NORMALIZED_FACADE_SHA256,
        "candidate_path": "src/falsewake/experiment_005_run_authority.py",
        "candidate_sha256": FACADE_SHA256,
        "normalized_added_lines": 1,
        "normalized_deleted_lines": 1,
        "reference_path": "src/falsewake/experiment_004_run_authority.py",
        "reference_sha256": P4_REFERENCE_SHA256,
        "status": "fail",
    }
    violation = report["violations"][0]
    assert violation["missing_substitution"] == {
        "candidate": "profile is ``005``",
        "reference": "profile is ``004``",
    }
    assert violation["candidate_normalized_sha256"] == NORMALIZED_FACADE_SHA256
    assert violation["candidate_sha256"] == FACADE_SHA256
    assert violation["reference_sha256"] == P4_REFERENCE_SHA256
    assert violation["normalized_added_lines"] == 1
    assert violation["normalized_deleted_lines"] == 1


def test_operational_facade_ast_remains_bound_only_to_profile_005() -> None:
    source = FACADE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(FACADE_PATH))
    imports = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert len(imports) == 2
    engine_import = imports[1]
    assert isinstance(engine_import, ast.ImportFrom)
    assert engine_import.module == "falsewake"
    assert len(engine_import.names) == 1
    assert engine_import.names[0].name == "experiment_002_run_authority"
    assert engine_import.names[0].asname == "_engine"

    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    expected_statements = {
        "verify_and_issue_experiment_005_run_registration": [
            "return _engine._verify_and_issue_experiment_005_run_registration()"
        ],
        "_verify_and_issue_experiment_005_sealed_child_registration": [
            (
                "return _engine."
                "_verify_and_issue_experiment_005_sealed_child_registration()"
            )
        ],
        "verify_verified_run_registration": [
            "_engine._verify_experiment_005_run_registration(registration)"
        ],
        "reverify_verified_run_registration": [
            "_engine._reverify_experiment_005_run_registration(registration)"
        ],
        "_registered_child_input_snapshot": [
            (
                "_engine._require_verified_registration_profile("
                "registration, _engine._EXPERIMENT_005_PROFILE)"
            ),
            "return _engine._registered_child_input_snapshot(registration)",
        ],
        "_create_sealed_experiment_005_child_bundle_fd": [
            (
                "return _engine."
                "_create_sealed_experiment_005_child_bundle_fd(registration)"
            )
        ],
        "_verified_state": [
            (
                "return _engine._require_verified_registration_profile("
                "registration, _engine._EXPERIMENT_005_PROFILE)"
            )
        ],
        "_required_child_bundle_seals": [
            "return _engine._required_child_bundle_seals()"
        ],
    }
    assert tuple(functions) == tuple(expected_statements)
    for name, expected in expected_statements.items():
        function = functions[name]
        assert [ast.unparse(node) for node in _without_docstring(function)] == (
            expected
        )
        assert all(argument.arg != "profile" for argument in function.args.args)
        assert all(argument.arg != "profile" for argument in function.args.posonlyargs)
        assert not function.args.kwonlyargs
        assert function.args.vararg is None
        assert function.args.kwarg is None

    assignments = {
        node.targets[0].id: ast.unparse(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    assert assignments["_SEALED_CHILD_MEMFD_TARGET"] == (
        "_engine._EXPERIMENT_005_SEALED_CHILD_MEMFD_TARGET"
    )
    assert "profile is ``005``" in source
    assert "_EXPERIMENT_004_PROFILE" not in source
    assert "_EXPERIMENT_006_PROFILE" not in source
    assert "experiment_005_coordinator" not in source
    assert "experiment_005_runner" not in source
    assert not os.path.lexists(P4_MARKER)
    assert not os.path.lexists(P5_MARKER)


def test_incident_requires_only_new_p6_recovery_and_forbids_p5_reuse() -> None:
    report = _report()
    assert tuple(report["recovery_requirements"]) == EXPECTED_RECOVERY_REQUIREMENTS
    assert report["preflight_evidence"]["full_registration_gate"] == {
        "reason": (
            "frozen_facade_normalization_cannot_express_the_truthful_"
            "profile_005_docstring"
        ),
        "status": "not_run",
    }
    requirements = "\n".join(report["recovery_requirements"])
    assert "Use only a new Experiment 006 identifier" in requirements
    assert "Freeze the corrected Experiment 006 protocol" in requirements
    assert "Do not create an Experiment 005 registration" in requirements
    assert "profile is ``006`` to profile is ``005``" in requirements
    assert "retry Experiment 005" not in requirements
    assert not os.path.lexists(ROOT / "configs/experiment-005-run.json")
    assert not os.path.lexists(P4_MARKER)
    assert not os.path.lexists(P5_MARKER)
