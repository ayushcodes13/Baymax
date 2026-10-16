"""Load the pinned public Baymax full model and run a fictional greeting smoke check."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path("/Volumes/PEN DRIVE/Project Work/ayushcodes13/Baymax").resolve()
if Path.cwd().resolve() != REPO_ROOT:
    raise SystemExit("Run from the Baymax checkout on the PEN DRIVE.")
if not (REPO_ROOT / ".venv-inference").is_dir():
    raise SystemExit("Create the local inference environment under the PEN DRIVE checkout first.")

config = json.loads((REPO_ROOT / "inference_config.json").read_text(encoding="utf-8"))
cache_dir = REPO_ROOT / config["cache_dir"]
cache_dir.mkdir(parents=True, exist_ok=True)
model_id = config["model"]["id"]
revision = config["model"]["revision"]

# This synthetic prompt checks load/generation mechanics only; it is not an evaluation case.
messages = [
    {"role": "system", "content": "For this software smoke check, reply with one brief greeting. Do not give health or medical guidance."},
    {"role": "user", "content": "Say hello to the Baymax local smoke check."},
]

tokenizer = AutoTokenizer.from_pretrained(
    model_id, revision=revision, cache_dir=str(cache_dir), trust_remote_code=False
)
model_dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    revision=revision,
    cache_dir=str(cache_dir),
    torch_dtype=model_dtype,
    use_safetensors=True,
    trust_remote_code=False,
)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
model.eval()
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {name: value.to(device) for name, value in inputs.items()}
with torch.inference_mode():
    generated = model.generate(
        **inputs,
        max_new_tokens=int(config["max_new_tokens"]),
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
text = tokenizer.decode(generated[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
if not text:
    raise RuntimeError("The model loaded but generated no visible text.")

try:
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()
except (OSError, subprocess.CalledProcessError):
    source_commit = None

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = {
    "status": "completed",
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_commit": source_commit,
    "model_id": model_id,
    "model_revision": revision,
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "device": device,
        "mps_available": bool(torch.backends.mps.is_available()),
        "torch_cuda_version": torch.version.cuda,
    },
    "source_files_sha256": {
        "inference_config.json": sha256(REPO_ROOT / "inference_config.json"),
        "requirements-inference.lock": sha256(REPO_ROOT / "requirements-inference.lock"),
        "scripts/inference_smoke.py": sha256(REPO_ROOT / "scripts/inference_smoke.py"),
    },
    "synthetic_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    "output_character_count": len(text),
    "note": "Load/generation smoke only; this result is not a model quality, safety, or medical evaluation.",
}
manifest_path = REPO_ROOT / config["manifest_path"]
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": "completed", "device": device, "model_id": model_id, "model_revision": revision, "generated_text": text}, indent=2))
