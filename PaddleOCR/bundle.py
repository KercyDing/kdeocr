#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=0.30", "onnx>=1.17", "PyYAML>=6.0"]
# ///

"""Build a PP-OCRv6 model release bundle."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
from pathlib import Path
from urllib.parse import urlparse

import onnx
import yaml
from huggingface_hub import hf_hub_download, model_info


class Profile:
    def __init__(self, profile_name: str, det_url: str, rec_url: str) -> None:
        self.profile_name = profile_name
        self.det_url = det_url
        self.rec_url = rec_url

    @property
    def archive_name(self) -> str:
        return f"kdeocr-{self.profile_name}"


def load_profiles(project_root: Path) -> dict[str, Profile]:
    index_path = project_root.parent / "index.toml"
    document = tomllib.loads(index_path.read_text(encoding="utf-8"))
    profiles = document.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError(f"{index_path}: missing profiles table")
    result = {}
    for profile_name, values in profiles.items():
        if not isinstance(profile_name, str) or not isinstance(values, dict):
            raise ValueError(f"{index_path}: invalid profile entry")
        try:
            det_url = values["det"]
            rec_url = values["rec"]
        except KeyError as error:
            raise ValueError(f"{index_path}: profile {profile_name} is missing {error.args[0]}") from error
        if not isinstance(det_url, str) or not isinstance(rec_url, str):
            raise ValueError(f"{index_path}: profile {profile_name} has invalid model URLs")
        result[profile_name] = Profile(profile_name, det_url, rec_url)
    return result


def repository_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "huggingface.co":
        raise ValueError(f"unsupported Hugging Face URL: {url}")
    parts = parsed.path.strip("/").split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"invalid Hugging Face repository URL: {url}")
    return "/".join(parts)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_shape(value: onnx.ValueInfoProto) -> list[int]:
    shape = value.type.tensor_type.shape
    return [dimension.dim_value if dimension.dim_value else -1 for dimension in shape.dim]


def inspect_model(path: Path, expected_shape: list[int], expected_classes: int | None = None) -> tuple[str, list[int], str, list[int]]:
    model = onnx.load(path.as_posix())
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError(f"{path}: expected one input and one output")
    input_value = model.graph.input[0]
    output_value = model.graph.output[0]
    input_shape = tensor_shape(input_value)
    output_shape = tensor_shape(output_value)
    if input_shape != expected_shape:
        raise ValueError(f"{path}: unexpected input shape {input_shape}")
    if expected_classes is not None and output_shape[-1] != expected_classes:
        raise ValueError(f"{path}: expected {expected_classes} output classes, got {output_shape[-1]}")
    return input_value.name, input_shape, output_value.name, output_shape


def read_charset(path: Path) -> list[str]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    characters = document["PostProcess"]["character_dict"]
    if not isinstance(characters, list) or not all(isinstance(char, str) and len(char) == 1 for char in characters):
        raise ValueError(f"{path}: character_dict is not a list of single characters")
    return characters


def write_manifest(
    bundle: Path,
    det_input: str,
    det_output: str,
    rec_input: str,
    rec_output: str,
    det_hash: str,
    rec_hash: str,
    charset_hash: str,
    charset_count: int,
    profile: Profile,
    det_revision: str,
    rec_revision: str,
) -> None:
    base_name, revision = profile.profile_name.rsplit("-r", 1)
    manifest = f'''format = 1
profile = "{base_name}"
revision = {int(revision)}
license = "LICENSE-PaddleOCR"
charset = "charset.txt"
charset_sha256 = "{charset_hash}"
charset_count = {charset_count}
blank_index = 0
space_index = {charset_count + 1}

[det]
revision = "{det_revision}"
model = "det/inference.onnx"
sha256 = "{det_hash}"
input_name = "{det_input}"
output_name = "{det_output}"
input_shape = [-1, 3, -1, -1]
output_shape = [-1, 1, -1, -1]

[rec]
revision = "{rec_revision}"
model = "rec/inference.onnx"
sha256 = "{rec_hash}"
input_name = "{rec_input}"
output_name = "{rec_output}"
input_shape = [-1, 3, 48, -1]
output_shape = [-1, -1, {charset_count + 2}]
'''
    (bundle / "manifest.toml").write_text(manifest, encoding="utf-8")


def normalize_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mtime = 0
    info.mode = 0o755 if info.isdir() else 0o644
    return info


def download_models(profile: Profile, bundle: Path) -> tuple[Path, Path, Path, str, str]:
    det_dir = bundle / "det"
    rec_dir = bundle / "rec"
    det_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)

    det_repository = repository_id(profile.det_url)
    rec_repository = repository_id(profile.rec_url)
    det_revision = model_info(det_repository).sha
    rec_revision = model_info(rec_repository).sha
    if not det_revision or not rec_revision:
        raise ValueError("Hugging Face did not return model revisions")
    det_model = Path(hf_hub_download(det_repository, "inference.onnx", revision=det_revision))
    rec_model = Path(hf_hub_download(rec_repository, "inference.onnx", revision=rec_revision))
    rec_config = Path(hf_hub_download(rec_repository, "inference.yml", revision=rec_revision))

    det_target = det_dir / "inference.onnx"
    rec_target = rec_dir / "inference.onnx"
    shutil.copy2(det_model, det_target)
    shutil.copy2(rec_model, rec_target)
    return det_target, rec_target, rec_config, det_revision, rec_revision


def build(args: argparse.Namespace, profiles: dict[str, Profile]) -> None:
    project_root = Path(__file__).resolve().parent
    profile = profiles[args.profile_name]
    archive_name = profile.archive_name
    bundle = project_root / archive_name
    license_path = project_root / "LICENSE-PaddleOCR"
    if not license_path.is_file():
        raise ValueError(f"missing {license_path}")
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    det_model, rec_model, rec_config, det_revision, rec_revision = download_models(profile, bundle)

    det_hash = sha256(det_model)
    rec_hash = sha256(rec_model)

    det_input, _, det_output, _ = inspect_model(det_model, [-1, 3, -1, -1])
    characters = read_charset(rec_config)
    rec_input, _, rec_output, rec_shape = inspect_model(rec_model, [-1, 3, 48, -1], len(characters) + 2)
    if rec_shape[-1] != len(characters) + 2:
        raise ValueError("recognition output and character dictionary are inconsistent")

    shutil.copy2(license_path, bundle / "LICENSE-PaddleOCR")
    (bundle / "charset.txt").write_text("\n".join(characters) + "\n", encoding="utf-8")
    write_manifest(
        bundle,
        det_input,
        det_output,
        rec_input,
        rec_output,
        det_hash,
        rec_hash,
        sha256(bundle / "charset.txt"),
        len(characters),
        profile,
        det_revision,
        rec_revision,
    )

    with tempfile.TemporaryDirectory(prefix="kdeocr-model-") as temporary:
        tar_path = Path(temporary) / f"{archive_name}.tar"
        with tarfile.open(tar_path, "w") as archive:
            for relative in ("LICENSE-PaddleOCR", "charset.txt", "manifest.toml", "det", "rec"):
                archive.add(bundle / relative, arcname=f"{archive_name}/{relative}", filter=normalize_tar_info)
        output = project_root / f"{archive_name}.tar.zst"
        subprocess.run(["zstd", "--ultra", "-22", "-T0", "-f", str(tar_path), "-o", str(output)], check=True)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    profiles = load_profiles(project_root)
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", dest="profile_name", required=True, choices=profiles)
    build(parser.parse_args(), profiles)


if __name__ == "__main__":
    main()
