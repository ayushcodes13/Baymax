"""Remove AppleDouble sidecars that break Python in the PEN DRIVE's FAT filesystem."""
from pathlib import Path

REPO_ROOT = Path("/Volumes/PEN DRIVE/Project Work/ayushcodes13/Baymax").resolve()
ENV_ROOT = REPO_ROOT / ".venv-inference"
if Path.cwd().resolve() != REPO_ROOT or not ENV_ROOT.is_dir():
    raise SystemExit("Run from the Baymax checkout on the PEN DRIVE after creating .venv-inference.")

removed = 0
for path in ENV_ROOT.rglob("._*"):
    if path.is_file() and not path.is_symlink():
        try:
            path.unlink()
            removed += 1
        except FileNotFoundError:
            pass
print(f"Removed {removed} AppleDouble sidecar files from .venv-inference.")
