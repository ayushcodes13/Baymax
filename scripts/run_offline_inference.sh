#!/usr/bin/env bash
# Run the pinned, cached model without network access. Keep every artifact on the PEN DRIVE.
set -u
repo='/Volumes/PEN DRIVE/Project Work/ayushcodes13/Baymax'
if [[ "${PWD}" != "${repo}" || ! -x .venv-inference/bin/python ]]; then
  echo 'Run from the Baymax checkout on PEN DRIVE with .venv-inference installed.' >&2
  exit 2
fi
if ! mkdir -p .tmp .cache/xdg .cache/torch .cache/pip .cache/huggingface; then
  echo 'Could not prepare PEN DRIVE temporary and cache directories.' >&2
  exit 2
fi
export TMPDIR="${repo}/.tmp/"
export XDG_CACHE_HOME="${repo}/.cache/xdg"
export PIP_CACHE_DIR="${repo}/.cache/pip"
export TORCH_HOME="${repo}/.cache/torch"
export HF_HOME="${repo}/.cache/huggingface"
export HF_HUB_CACHE="${repo}/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
run_id="$( .venv-inference/bin/python -B -c 'import uuid; print(uuid.uuid4().hex)' )" || {
  echo 'Could not generate a unique inference run ID.' >&2
  exit 2
}
if [[ ! "$run_id" =~ ^[0-9a-f]{32}$ ]]; then
  echo 'Invalid inference run ID.' >&2
  exit 2
fi
export BAYMAX_RUN_ID="$run_id"
run_log=".tmp/offline-inference-${run_id}.log"
run_result=".tmp/offline-inference-${run_id}.exit"
run_manifest=".tmp/offline-inference-${run_id}.manifest.json"
printf 'Offline inference log: %s\n' "$run_log"
printf 'Offline inference manifest: %s\n' "$run_manifest"
set -o pipefail
.venv-inference/bin/python scripts/inference_smoke.py 2>&1 | tee "$run_log"
pipeline_status=( "${PIPESTATUS[@]}" )
code="${pipeline_status[0]}"
if (( pipeline_status[1] != 0 )); then
  printf 'Could not capture the complete inference log in %s (tee exit %s).\n' "$run_log" "${pipeline_status[1]}" >&2
  if (( code == 0 )); then
    code="${pipeline_status[1]}"
  fi
fi
if (( code == 0 )) && [[ ! -s "$run_manifest" ]]; then
  echo 'Inference reported success but its per-run manifest is missing or empty.' >&2
  code=4
fi
exit_record_tmp="${run_result}.tmp"
exit_timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)" || exit 3
if ! printf '{"run_id":"%s","exit_code":%s,"recorded_at_utc":"%s"}\n' "$run_id" "$code" "$exit_timestamp" > "$exit_record_tmp"; then
  printf 'Could not record inference exit status in %s.\n' "$run_result" >&2
  exit 3
fi
if ! mv "$exit_record_tmp" "$run_result"; then
  printf 'Could not finalize inference exit status in %s.\n' "$run_result" >&2
  exit 3
fi
printf 'Offline inference exit status: %s (recorded in %s)\n' "$code" "$run_result"
exit "$code"
