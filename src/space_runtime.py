"""Runtime helpers used only by the hosted Hugging Face Space."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from zipfile import ZipFile


LANGUAGE_MODEL_REPO = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
LANGUAGE_MODEL_REVISION = "69e9000591623e5a4fc2f502407860bcdc0de0b2"
LANGUAGE_MODEL_ARCHIVE = "vi_lm_4grams.bin.zip"
LANGUAGE_MODEL_MEMBER = "vi_lm_4grams.bin"


def extract_language_model(
    archive_path: str | Path,
    target_path: str | Path,
) -> Path:
    """Atomically extract exactly the expected LM member, never arbitrary paths."""
    archive = Path(archive_path)
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with ZipFile(archive) as zip_file:
            if LANGUAGE_MODEL_MEMBER not in zip_file.namelist():
                raise ValueError(
                    f"Archive does not contain {LANGUAGE_MODEL_MEMBER!r}."
                )
            with zip_file.open(LANGUAGE_MODEL_MEMBER) as source, temporary.open(
                "wb"
            ) as destination:
                shutil.copyfileobj(source, destination)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def ensure_language_model(target_path: str | Path) -> Path:
    """Download the official model archive through the Hub cache when needed."""
    target = Path(target_path)
    if target.is_file() and target.stat().st_size > 0:
        return target

    from huggingface_hub import hf_hub_download

    archive = hf_hub_download(
        repo_id=LANGUAGE_MODEL_REPO,
        filename=LANGUAGE_MODEL_ARCHIVE,
        revision=LANGUAGE_MODEL_REVISION,
    )
    return extract_language_model(archive, target)
