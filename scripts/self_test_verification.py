"""No-download Baymax verifier and wrapper regression suite.

Runs exclusively with the Python standard library and synthetic .tmp fixtures.
A passing test NEVER proves a new model inference or the original historical run.
"""
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
import uuid

from verify_inference_manifest import EvidenceError, verify_recorded_run, RESULT_PREFIX


CHECKOUT = Path(__file__).resolve().parent.parent
TEMP_ROOT = CHECKOUT / ".tmp"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)
assert TEMP_ROOT.resolve().is_relative_to(Path("/Volumes/PEN DRIVE").resolve()), (
    "The self-test must run from a checkout on PEN DRIVE."
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class VerificationRegression(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="baymax-source-self-test-", dir=TEMP_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "fixture-checkout"
        self.root.mkdir()
        (self.root / ".tmp").mkdir()
        (self.root / "scripts").mkdir()
        (self.root / "scripts/inference_smoke.py").write_text("synthetic source file\n")
        (self.root / "requirements-inference.lock").write_text("synthetic lock\n")
        self.config = {
            "model": {"id": "test/no-download-model", "revision": "0123456789abcdef"}
        }
        write_json(self.root / "inference_config.json", self.config)
        self.id = "a" * 32
        self.output = "a synthetic hello"
        self.manifest = {
            "status": "completed",
            "run_id": self.id,
            "recorded_at_utc": "2026-10-14T20:00:22.900000+00:00",
            "model_id": self.config["model"]["id"],
            "model_revision": self.config["model"]["revision"],
            "output_character_count": len(self.output),
            "output_sha256": hashlib.sha256(self.output.encode()).hexdigest(),
            "source_files_sha256": {
                name: hashlib.sha256((self.root / name).read_bytes()).hexdigest()
                for name in ("inference_config.json", "requirements-inference.lock",
                             "scripts/inference_smoke.py")
            }
        }
        self.exit_record = {
            "run_id": self.id,
            "exit_code": 0,
            "recorded_at_utc": "2026-10-14T20:00:23+00:00"
        }
        self.log = self.root / ".tmp" / ("offline-inference-" + self.id + ".log")
        self.exit = self.log.with_suffix(".exit")
        self.mfile = self.log.with_suffix(".manifest.json")
        self.refresh()

    def refresh(self, frames=None, warning="library warning", trailing=None):
        write_json(self.mfile, self.manifest)
        write_json(self.exit, self.exit_record)
        raw = self.mfile.read_bytes()
        frame = {
            "frame_version": 1,
            "run_id": self.id,
            "status": "completed",
            "recorded_at_utc": self.manifest["recorded_at_utc"],
            "model_id": self.manifest["model_id"],
            "model_revision": self.manifest["model_revision"],
            "generated_text": self.output,
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        }
        self.frame = frame
        lines = [warning]
        for payload in ([frame] if frames is None else frames):
            lines.append(RESULT_PREFIX + (payload if isinstance(payload, str)
                                          else json.dumps(payload, separators=(",", ":"))))
        if trailing:
            lines.append(trailing)
        self.log.write_text("\n".join(lines) + "\n")

    def pass_check(self):
        self.assertEqual(verify_recorded_run(self.root, self.log, self.exit), self.id)

    def fail_check(self, pattern):
        with self.assertRaisesRegex(EvidenceError, pattern):
            verify_recorded_run(self.root, self.log, self.exit)

    def test_01_valid_synthetic_pair(self):
        self.pass_check()

    def test_02_old_manifest_identical_output_with_new_run_id(self):
        self.manifest["run_id"] = "b" * 32
        self.refresh()
        self.fail_check("manifest run_id mismatch")

    def test_03_swapped_exit_run_id(self):
        self.exit_record["run_id"] = "b" * 32
        self.refresh()
        self.fail_check("exit run_id mismatch")

    def test_04_swapped_result_frame_run_id(self):
        self.refresh()
        self.frame["run_id"] = "b" * 32
        self.refresh(frames=[self.frame])
        self.fail_check("result frame run_id mismatch")

    def test_05_duplicate_final_result_frame(self):
        self.refresh()
        self.refresh(frames=[self.frame, self.frame])
        self.fail_check("exactly one marked result frame")

    def test_06_malformed_result_frame(self):
        self.refresh(frames=['{"run_id":'])
        self.fail_check("result frame: invalid JSON")

    def test_07_unmarked_lookalike_json_does_not_pass(self):
        self.refresh(frames=[])
        self.log.write_text(json.dumps(self.frame) + "\n")
        self.fail_check("exactly one marked result frame")

    def test_08_unexpected_trailing_log_line(self):
        self.refresh(trailing="late output after result")
        self.fail_check("final log line")

    def test_09_missing_exit_record(self):
        self.exit.unlink()
        self.fail_check("exit record: missing")

    def test_10_nonzero_exit_record(self):
        self.exit_record["exit_code"] = 7
        self.refresh()
        self.fail_check("not zero")

    def test_11_missing_manifest(self):
        self.mfile.unlink()
        self.fail_check("per-run manifest: missing")

    def test_12_tampered_generated_text(self):
        self.refresh()
        self.frame["generated_text"] = "different output"
        self.refresh(frames=[self.frame])
        self.fail_check("character count mismatch")

    def test_13_source_mismatch(self):
        (self.root / "scripts/inference_smoke.py").write_text("modified synthetic source\n")
        self.fail_check("source hash mismatch")

    def test_14_wrong_model_revision(self):
        self.config["model"]["revision"] = "not-pinned"
        write_json(self.root / "inference_config.json", self.config)
        self.manifest["source_files_sha256"]["inference_config.json"] = hashlib.sha256(
            (self.root / "inference_config.json").read_bytes()).hexdigest()
        self.refresh()
        self.fail_check("model revision mismatch")

    def test_15_manifest_digest_tampering(self):
        self.refresh()
        self.frame["manifest_sha256"] = "0" * 64
        self.refresh(frames=[self.frame])
        self.fail_check("manifest bytes")

    def test_16_malformed_exit_json(self):
        self.exit.write_text('{"run_id":')
        self.fail_check("exit record: invalid JSON")

    def test_17_duplicate_exit_json_key(self):
        self.exit.write_text('{"run_id":"a","run_id":"b","exit_code":0}')
        self.fail_check("duplicate JSON key")

    def test_18_path_escape(self):
        with self.assertRaisesRegex(EvidenceError, "beneath"):
            verify_recorded_run(self.root, self.root / "inference_config.json",
                                self.root / "inference_config.exit")

    def test_19_symlink_escape(self):
        link = self.root / ".tmp" / ("offline-inference-" + "c" * 32 + ".log")
        link.symlink_to(self.root / "inference_config.json")
        with self.assertRaisesRegex(EvidenceError, "beneath"):
            verify_recorded_run(self.root, link, link.with_suffix(".exit"))

    def test_20_exit_timestamp_reversed(self):
        self.exit_record["recorded_at_utc"] = "2026-10-14T19:58:20+00:00"
        self.refresh()
        self.fail_check("predates the manifest")

    def test_21_timestamp_malformed(self):
        self.manifest["recorded_at_utc"] = "not ISO"
        self.refresh()
        self.fail_check("invalid ISO timestamp")

    def test_22_exit_and_log_filename_mismatch(self):
        with self.assertRaisesRegex(EvidenceError, "exact run ID"):
            verify_recorded_run(self.root, self.log,
                                self.root / ".tmp" / ("offline-inference-" + "b" * 32 + ".exit"))

    def test_23_missing_marked_frame_after_warning(self):
        self.refresh(frames=[], warning='{"status":"completed"}')
        self.fail_check("exactly one marked result frame")

    def test_24_boolean_exit_code_is_not_integer_zero(self):
        self.exit_record["exit_code"] = False
        self.refresh()
        self.fail_check("not zero")


class WrapperRegression(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="baymax-wrapper-self-test-", dir=TEMP_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "mock-checkout"
        (self.repo / ".tmp").mkdir(parents=True)
        (self.repo / ".cache").mkdir()
        self.bin = self.base / "bin"
        self.bin.mkdir()
        self.fake = self.base / "mock-inference.sh"
        self.fake.write_text(
            '#!/bin/bash\n'
            'if test "$BAYMAX_TEST_INFERENCE_EXIT" = 0; then\n'
            '  printf "synthetic-only\\n" > ".tmp/offline-inference-$BAYMAX_RUN_ID.manifest.json"\n'
            'fi\n'
            'echo "synthetic stub: no model invoked"\n'
            'exit "$BAYMAX_TEST_INFERENCE_EXIT"\n'
        )
        shim = self.bin / "tee"
        shim.write_text(
            '#!/bin/bash\n'
            'if test "$BAYMAX_TEST_TEE_FAIL" = yes; then\n'
            '  cat >/dev/null\n'
            '  echo "synthetic tee failure" >&2\n'
            '  exit 42\n'
            'fi\n'
            'exec /usr/bin/tee "$@"\n'
        )
        shim.chmod(0o700)
        original = (CHECKOUT / "scripts/run_offline_inference.sh").read_text()
        original = original.replace(
            "/Volumes/PEN DRIVE/Project Work/ayushcodes13/Baymax", str(self.repo)
        )
        original = original.replace(
            "! -x .venv-inference/bin/python", "! -x /usr/bin/python3"
        )
        original = original.replace(
            ".venv-inference/bin/python -B -c", "/usr/bin/python3 -B -c"
        )
        original = original.replace(
            ".venv-inference/bin/python scripts/inference_smoke.py",
            "bash " + shlex.quote(str(self.fake)),
        )
        assert "scripts/inference_smoke.py" not in original
        assert "repo='" + str(self.repo) + "'" in original
        self.original = original

    def run_stub(self, inference_rc, tee_fail=False, bad_exit=False, bad_rename=False):
        script = self.original
        if bad_exit:
            script, count = re.subn(
                r'^run_result=.*$', 'run_result=".tmp/absent-parent/invalid.exit"',
                script, flags=re.M
            )
            self.assertEqual(count, 1)
        if bad_rename:
            shim = self.bin / "mv"
            shim.write_text('#!/bin/bash\necho "synthetic rename failure" >&2\nexit 43\n')
            shim.chmod(0o700)
        harness = self.base / "wrapper-harness.sh"
        harness.write_text(script)
        env = os.environ.copy()
        env.update(
            PATH=str(self.bin) + os.pathsep + env.get("PATH", ""),
            TMPDIR=str(self.repo / ".tmp"),
            PYTHONDONTWRITEBYTECODE="1",
            BAYMAX_TEST_INFERENCE_EXIT=str(inference_rc),
            BAYMAX_TEST_TEE_FAIL="yes" if tee_fail else "no"
        )
        parsed = subprocess.run(["/bin/bash", "-n", str(harness)], capture_output=True, text=True, env=env)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        result = subprocess.run(["/bin/bash", str(harness)], cwd=self.repo,
                                capture_output=True, text=True, env=env)
        match = re.search(r"Offline inference log: (\S+)", result.stdout)
        self.assertIsNotNone(match, result.stdout)
        log = self.repo / match.group(1)
        return result, log.with_suffix(".exit")

    def test_25_wrapper_success(self):
        result, path = self.run_stub(0)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(path.read_text())["exit_code"], 0)

    def test_26_wrapper_inference_failure(self):
        result, path = self.run_stub(17)
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertEqual(json.loads(path.read_text())["exit_code"], 17)

    def test_27_wrapper_tee_failure(self):
        result, path = self.run_stub(0, tee_fail=True)
        self.assertEqual(result.returncode, 42, result.stderr)
        self.assertEqual(json.loads(path.read_text())["exit_code"], 42)
        self.assertIn("Could not capture", result.stderr)

    def test_28_wrapper_exit_write_failure(self):
        result, path = self.run_stub(0, bad_exit=True)
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertIn("Could not record", result.stderr)

    def test_29_wrapper_exit_rename_failure(self):
        result, path = self.run_stub(0, bad_rename=True)
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertIn("Could not finalize", result.stderr)

    def test_30_wrapper_missing_manifest(self):
        result, path = self.run_stub(18)
        self.assertEqual(result.returncode, 18)
        # Missing manifest on a nominally successful runner should become 4.
        fake_text = self.fake.read_text()
        self.fake.write_text(fake_text.replace('printf "synthetic-only\\n"', 'false && printf "synthetic-only\\n"'))
        result, path = self.run_stub(0)
        self.assertEqual(result.returncode, 4, result.stderr)
        self.assertEqual(json.loads(path.read_text())["exit_code"], 4)


if __name__ == "__main__":
    print("SOURCE-ONLY SELF-TEST: synthetic fixtures; no model loading, no model weights, no network.")
    result = unittest.main(verbosity=2, exit=False).result
    print("SELF-TEST ONLY: this does not corroborate a new or historical inference.")
    raise SystemExit(0 if result.wasSuccessful() else 1)
