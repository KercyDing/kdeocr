#!/usr/bin/env python3

"""Generate NOTICE for Rust runtime dependencies."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
NOTICE = ROOT / "NOTICE"


def run_cargo_license() -> list[dict[str, object]]:
    check = subprocess.run(
        ["cargo", "license", "--help"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if check.returncode != 0:
        print("cargo-license is not installed; installing it now...")
        subprocess.run(["cargo", "install", "cargo-license"], cwd=ROOT, check=True)

    output = subprocess.run(
        ["cargo", "license", "--json", "--avoid-dev-deps"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(output.stdout)
    if not isinstance(value, list):
        raise ValueError("cargo-license returned an invalid JSON document")
    return value


def license_text(crate: dict[str, object]) -> str | None:
    name = crate.get("name")
    version = crate.get("version")
    if not isinstance(name, str) or not isinstance(version, str):
        return None

    license_file = crate.get("license_file")
    if isinstance(license_file, str):
        path = Path(license_file)
        if path.is_file():
            return path.read_text(encoding="utf-8-sig").strip()

    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
    registry = cargo_home / "registry" / "src"
    if not registry.is_dir():
        return None
    for source in registry.iterdir():
        directory = source / f"{name}-{version}"
        if not directory.is_dir():
            continue
        candidates = sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file() and path.name.lower().startswith(("license", "copying"))
            ),
            key=lambda path: path.name,
        )
        if candidates:
            return candidates[0].read_text(encoding="utf-8-sig").strip()
    return None


def authors(crate: dict[str, object]) -> str | None:
    value = crate.get("authors")
    if isinstance(value, str) and value:
        return value.replace("|", ", ")
    if isinstance(value, list):
        values = [author for author in value if isinstance(author, str)]
        if values:
            return ", ".join(values)
    return None


def section(crate: dict[str, object], number: int) -> list[str]:
    name = crate["name"]
    version = crate["version"]
    if not isinstance(name, str) or not isinstance(version, str):
        raise ValueError("cargo-license returned a dependency without name or version")

    lines = [f"{number}. {name}@{version}", ""]
    crate_authors = authors(crate)
    if crate_authors:
        lines.append(f"Authors: {crate_authors}")
    repository = crate.get("repository")
    if isinstance(repository, str) and repository:
        lines.append(f"Repository: {repository}")
    license_name = crate.get("license")
    lines.append(f"License: {license_name if isinstance(license_name, str) else 'Unknown'}")
    lines.append("")
    text = license_text(crate)
    if text:
        lines.extend([text, ""])
    return lines


def main() -> None:
    crates = run_cargo_license()
    dependencies = sorted(
        (
            crate
            for crate in crates
            if crate.get("name") != "kdeocr"
        ),
        key=lambda crate: (str(crate.get("name", "")), str(crate.get("version", ""))),
    )
    lines = [
        "KOCR",
        "Licensed under the Apache License, Version 2.0.",
        "",
        "---",
        "",
        "Third-party Rust dependencies:",
        "",
    ]
    for number, crate in enumerate(dependencies, start=1):
        lines.extend(section(crate, number))
    NOTICE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {NOTICE}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"notice: {error}", file=sys.stderr)
        raise SystemExit(1) from error
