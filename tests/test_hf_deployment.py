from __future__ import annotations

from pathlib import Path, PurePosixPath
import subprocess
import sys
from zipfile import ZipFile

import pytest

from src.hf_deployment import build_space_manifest
from src.space_runtime import ensure_language_model, extract_language_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_deployment_script_runs_directly_from_the_project_root() -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/deploy_hf_space.py"), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--repo-id" in result.stdout


def test_space_manifest_is_a_small_explicit_inference_only_allowlist() -> None:
    manifest = build_space_manifest(PROJECT_ROOT)
    remote_paths = [entry.path_in_repo for entry in manifest]

    assert {
        "README.md",
        "app.py",
        "requirements.txt",
        "packages.txt",
        "config.yaml",
        "app/gradio_app.py",
        "src/inference.py",
        "src/space_runtime.py",
        "checkpoints/phowhisper_st23d_spoken_code_adaptation/checkpoint-16/model.safetensors",
    }.issubset(remote_paths)
    assert len(remote_paths) == len(set(remote_paths))
    assert all(entry.local_path.is_file() for entry in manifest)
    assert all(not PurePosixPath(path).is_absolute() for path in remote_paths)

    forbidden_names = {
        ".env",
        "optimizer.pt",
        "rng_state.pth",
        "scaler.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
        "vi_lm_4grams.bin",
        "vi_lm_4grams.bin.zip",
    }
    assert not forbidden_names.intersection(PurePosixPath(path).name for path in remote_paths)
    assert not any(path.startswith(("data/", "outputs/", "tmp/", "models/")) for path in remote_paths)

    payload_bytes = sum(entry.local_path.stat().st_size for entry in manifest)
    assert payload_bytes < 400 * 1024 * 1024


def test_extract_language_model_copies_only_the_expected_zip_member(tmp_path: Path) -> None:
    archive = tmp_path / "lm.zip"
    target = tmp_path / "models" / "vi_lm_4grams.bin"
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("vi_lm_4grams.bin", b"language-model")
        zip_file.writestr("../outside.txt", b"must-not-be-written")

    result = extract_language_model(archive, target)

    assert result == target
    assert target.read_bytes() == b"language-model"
    assert not (tmp_path / "outside.txt").exists()


def test_extract_language_model_rejects_archive_without_expected_member(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "malicious.zip"
    target = tmp_path / "models" / "vi_lm_4grams.bin"
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("../vi_lm_4grams.bin", b"wrong-member")

    with pytest.raises(ValueError, match="vi_lm_4grams.bin"):
        extract_language_model(archive, target)

    assert not target.exists()


def test_language_model_download_is_pinned_to_an_immutable_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "lm.zip"
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("vi_lm_4grams.bin", b"language-model")
    calls: list[dict[str, str]] = []

    def fake_download(**kwargs: str) -> str:
        calls.append(kwargs)
        return str(archive)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

    ensure_language_model(tmp_path / "model" / "vi_lm_4grams.bin")

    assert len(calls) == 1
    assert calls[0]["revision"] == "69e9000591623e5a4fc2f502407860bcdc0de0b2"
