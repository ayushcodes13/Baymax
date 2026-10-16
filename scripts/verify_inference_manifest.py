"""Verify one uniquely identified Baymax inference run using only Python's stdlib.

A PASS attests that a selected recorded run's manifest, final log frame and
exit record agree. It does not execute or evaluate a model.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re


RESULT_PREFIX = "BAYMAX_RESULT_V1="
RUN_PATTERN = re.compile(r"offline-inference-([0-9a-f]{32})\.log\Z")
SHA_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
SOURCE_FILES = ("inference_config.json", "requirements-inference.lock", "scripts/inference_smoke.py")


class EvidenceError(Exception):
    """The recorded evidence is missing, malformed, or inconsistent."""


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def load_json(data, label):
    try:
        item = json.loads(data, object_pairs_hook=strict_object)
    except (ValueError, UnicodeDecodeError) as exc:
        raise EvidenceError(f"{label}: invalid JSON ({exc})") from exc
    require(isinstance(item, dict), f"{label}: expected JSON object")
    return item


def read_small(path, limit, label):
    require(path.is_file(), f"{label}: missing file")
    try:
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
    except OSError as exc:
        raise EvidenceError(f"{label}: cannot read file ({exc})") from exc
    require(len(data) <= limit, f"{label}: file exceeds {limit} bytes")
    return data


def utc_timestamp(value, label):
    require(isinstance(value, str), f"{label}: missing ISO UTC timestamp")
    try:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise EvidenceError(f"{label}: invalid ISO timestamp") from exc
    require(parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0),
            f"{label}: timestamp must specify UTC")
    return parsed


def verify_recorded_run(root, run_log, exit_file):
    """Validate run-specific evidence. Safe to call with synthetic fixture roots."""
    root = Path(root).resolve()
    temp_root = (root / ".tmp").resolve()
    log = (root / run_log).resolve()
    status_file = (root / exit_file).resolve()
    require(log.is_relative_to(temp_root) and status_file.is_relative_to(temp_root),
            "run evidence must resolve beneath this checkout's .tmp")
    match = RUN_PATTERN.fullmatch(log.name)
    require(match is not None, "log must have a unique 32-hex run ID in its filename")
    run_id = match.group(1)
    require(status_file == log.with_suffix(".exit"),
            "exit record must share the selected log's exact run ID")
    manifest_file = log.with_suffix(".manifest.json")

    config = load_json(read_small(root / "inference_config.json", 65536, "config"), "config")
    manifest_data = read_small(manifest_file, 65536, "per-run manifest")
    manifest = load_json(manifest_data, "per-run manifest")
    exit_record = load_json(read_small(status_file, 4096, "exit record"), "exit record")

    require(manifest.get("run_id") == run_id, "manifest run_id mismatch")
    require(exit_record.get("run_id") == run_id, "exit run_id mismatch")
    require(manifest.get("status") == "completed", "manifest is not completed")
    require(type(exit_record.get("exit_code")) is int and exit_record["exit_code"] == 0,
            "recorded inference exit status is not zero")
    require(manifest.get("model_id") == config["model"]["id"], "model ID mismatch")
    require(manifest.get("model_revision") == config["model"]["revision"],
            "model revision mismatch")
    require(type(manifest.get("output_character_count")) is int
            and manifest["output_character_count"] > 0, "empty output count")
    require(isinstance(manifest.get("output_sha256"), str)
            and SHA_PATTERN.fullmatch(manifest["output_sha256"]) is not None,
            "invalid output SHA256")

    hashes = manifest.get("source_files_sha256")
    require(isinstance(hashes, dict), "missing source file hashes")
    for name in SOURCE_FILES:
        path = root / name
        require(path.is_file(), f"missing source file: {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(hashes.get(name) == actual, f"source hash mismatch: {name}")

    manifest_time = utc_timestamp(manifest.get("recorded_at_utc"), "manifest")
    exit_time = utc_timestamp(exit_record.get("recorded_at_utc"), "exit record")
    # The manifest carries microseconds, while the wrapper's date command writes seconds.
    # Compare recorded timestamps, never unreliable removable-FAT file mtimes.
    require(exit_time >= manifest_time - timedelta(seconds=2),
            "exit record predates the manifest by over 2 seconds")

    lines = read_small(log, 1048576, "run log").decode("utf-8", errors="replace").splitlines()
    frames = [line[len(RESULT_PREFIX):] for line in lines if line.startswith(RESULT_PREFIX)]
    require(len(frames) == 1, "run log must contain exactly one marked result frame")
    require(lines and lines[-1].startswith(RESULT_PREFIX),
            "marked result frame must be the final log line")
    frame = load_json(frames[0], "result frame")
    require(type(frame.get("frame_version")) is int and frame["frame_version"] == 1,
            "unsupported result frame version")
    require(frame.get("run_id") == run_id, "result frame run_id mismatch")
    require(frame.get("status") == "completed", "result frame is not completed")
    require(frame.get("recorded_at_utc") == manifest.get("recorded_at_utc"),
            "result timestamp differs from manifest")
    require(frame.get("model_id") == manifest.get("model_id"),
            "result model ID mismatch")
    require(frame.get("model_revision") == manifest.get("model_revision"),
            "result model revision mismatch")
    require(frame.get("manifest_sha256") == hashlib.sha256(manifest_data).hexdigest(),
            "result does not identify the selected manifest bytes")
    output = frame.get("generated_text")
    require(isinstance(output, str) and bool(output.strip()), "empty result text")
    require(len(output) == manifest["output_character_count"],
            "result text character count mismatch")
    require(hashlib.sha256(output.encode("utf-8")).hexdigest() == manifest["output_sha256"],
            "result text SHA256 mismatch")
    return run_id


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-log", type=Path, required=True, help="Per-run .log under .tmp/")
    parser.add_argument("--exit-status", type=Path, required=True,
                        help="Matching per-run .exit JSON under .tmp/")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    if Path.cwd().resolve() != root:
        parser.error("Run this verifier from its Baymax checkout root")
    try:
        run_id = verify_recorded_run(root, args.run_log, args.exit_status)
    except (EvidenceError, KeyError, OSError, UnicodeDecodeError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"PASS: corroborated Baymax recorded run {run_id} (no model executed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
