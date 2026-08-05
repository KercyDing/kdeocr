#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=0.30", "onnx>=1.17", "PyYAML>=6.0"]
# ///

"""Build the reproducible PP-OCRv6-small model release bundle."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import onnx
import yaml
from huggingface_hub import hf_hub_download, model_info


DET_REPOSITORY = "PaddlePaddle/PP-OCRv6_small_det_onnx"
REC_REPOSITORY = "PaddlePaddle/PP-OCRv6_small_rec_onnx"


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
    det_revision: str,
    rec_revision: str,
) -> None:
    manifest = f'''format = 1
profile = "ppocrv6-small"
revision = 1
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


def download_models(bundle: Path) -> tuple[Path, Path, Path, str, str]:
    det_dir = bundle / "det"
    rec_dir = bundle / "rec"
    det_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)

    det_revision = model_info(DET_REPOSITORY).sha
    rec_revision = model_info(REC_REPOSITORY).sha
    if not det_revision or not rec_revision:
        raise ValueError("Hugging Face did not return model revisions")
    det_model = Path(hf_hub_download(DET_REPOSITORY, "inference.onnx", revision=det_revision))
    rec_model = Path(hf_hub_download(REC_REPOSITORY, "inference.onnx", revision=rec_revision))
    rec_config = Path(hf_hub_download(REC_REPOSITORY, "inference.yml", revision=rec_revision))

    det_target = det_dir / "inference.onnx"
    rec_target = rec_dir / "inference.onnx"
    shutil.copy2(det_model, det_target)
    shutil.copy2(rec_model, rec_target)
    return det_target, rec_target, rec_config, det_revision, rec_revision


def build(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parent
    bundle = project_root / "temp"
    license_path = project_root / "LICENSE-PaddleOCR"
    if not license_path.is_file():
        raise ValueError(f"missing {license_path}")
    bundle.mkdir(parents=True, exist_ok=True)
    det_model, rec_model, rec_config, det_revision, rec_revision = download_models(bundle)

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
        det_revision,
        rec_revision,
    )

    with tempfile.TemporaryDirectory(prefix="kdeocr-model-") as temporary:
        tar_path = Path(temporary) / f"{args.archive_name}.tar"
        with tarfile.open(tar_path, "w") as archive:
            for relative in ("LICENSE-PaddleOCR", "charset.txt", "manifest.toml", "det", "rec"):
                archive.add(bundle / relative, arcname=f"{args.archive_name}/{relative}", filter=normalize_tar_info)
        output = project_root / f"{args.archive_name}.tar.zst"
        subprocess.run(["zstd", "--ultra", "-22", "-T0", "-f", str(tar_path), "-o", str(output)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-name", default="kdeocr-ppocrv6-small-r1")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
