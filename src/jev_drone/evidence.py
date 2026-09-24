"""Freeze importable source packages alongside a recorded experiment."""
import hashlib
from pathlib import Path
import shutil


def snapshot_sources(destination, *, experiments=False):
    package = Path(__file__).resolve().parent
    roots = {"jev_drone": package}
    if experiments:
        import experiments as research

        roots["experiments"] = Path(research.__file__).resolve().parent
    destination.mkdir(parents=True, exist_ok=True)
    for name, source in roots.items():
        shutil.copytree(source, destination / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return {str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(destination.rglob("*")) if path.is_file()}
