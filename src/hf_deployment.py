"""Explicit Hugging Face Space upload allowlist.

The project contains raw data, optimizer states, reports and temporary files that
must never be published with the inference demo.  Deployment therefore starts
from this small allowlist instead of walking the repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpaceUpload:
    local_path: Path
    path_in_repo: str


_DIRECT_FILES = {
    "deploy/huggingface-space/README.md": "README.md",
    "deploy/huggingface-space/app.py": "app.py",
    "deploy/huggingface-space/requirements.txt": "requirements.txt",
    "deploy/huggingface-space/packages.txt": "packages.txt",
    "config.yaml": "config.yaml",
    "app/__init__.py": "app/__init__.py",
    "app/gradio_app.py": "app/gradio_app.py",
    "src/__init__.py": "src/__init__.py",
    "src/cli.py": "src/cli.py",
    "src/config.py": "src/config.py",
    "src/inference.py": "src/inference.py",
    "src/metrics.py": "src/metrics.py",
    "src/model_utils.py": "src/model_utils.py",
    "src/preprocessing.py": "src/preprocessing.py",
    "src/space_runtime.py": "src/space_runtime.py",
}

_CHECKPOINT_ROOT = Path(
    "checkpoints/phowhisper_st23d_spoken_code_adaptation/checkpoint-16"
)
_CHECKPOINT_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def build_space_manifest(project_root: str | Path) -> tuple[SpaceUpload, ...]:
    """Return the complete, validated upload allowlist for the public Space."""
    root = Path(project_root).expanduser().resolve()
    uploads = [
        SpaceUpload(root / local_path, path_in_repo)
        for local_path, path_in_repo in _DIRECT_FILES.items()
    ]
    uploads.extend(
        SpaceUpload(
            root / _CHECKPOINT_ROOT / filename,
            (_CHECKPOINT_ROOT / filename).as_posix(),
        )
        for filename in _CHECKPOINT_FILES
    )

    missing = [str(item.local_path) for item in uploads if not item.local_path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required Hugging Face deployment files: " + ", ".join(missing)
        )
    return tuple(uploads)
