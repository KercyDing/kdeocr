#!/usr/bin/env python3

"""Show or update the package version in Cargo.toml."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CARGO_TOML = ROOT / "Cargo.toml"
SEMVER = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)


def package_version(source: str, section_start: str, section_end: str, name: str) -> str:
    sections = re.findall(
        rf"(?ms)^{re.escape(section_start)}\n(.*?)(?=^{re.escape(section_end)}|\Z)", source
    )
    matches = []
    for section in sections:
        name_match = re.search(r'^name\s*=\s*"([^"]+)"\s*$', section, re.MULTILINE)
        version_match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', section, re.MULTILINE)
        if name_match and name_match.group(1) == name and version_match:
            matches.append(version_match.group(1))
    if len(matches) != 1:
        raise ValueError(f"expected one {name} package version")
    return matches[0]


def replace_package_version(
    source: str, section_start: str, section_end: str, name: str, version: str
) -> str:
    pattern = re.compile(
        rf"(?ms)^(?P<header>{re.escape(section_start)})\n"
        rf"(?P<section>.*?)(?=^{re.escape(section_end)}|\Z)"
    )
    updated = 0

    def replace_section(match: re.Match[str]) -> str:
        nonlocal updated
        section = match.group("section")
        name_match = re.search(r'^name\s*=\s*"([^"]+)"\s*$', section, re.MULTILINE)
        if not name_match or name_match.group(1) != name:
            return match.group(0)
        section, count = re.subn(
            r'^version\s*=\s*"[^"]+"\s*$',
            f'version = "{version}"',
            section,
            count=1,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise ValueError(f"{name} package is missing a version")
        updated += 1
        return f"{match.group('header')}\n{section}"

    result = pattern.sub(replace_section, source)
    if updated != 1:
        raise ValueError(f"expected one {name} package version")
    return result


def read_version() -> str:
    cargo_toml = CARGO_TOML.read_text(encoding="utf-8")
    return package_version(cargo_toml, "[package]", "[", "kdeocr")


def write_version(version: str) -> None:
    cargo_toml = CARGO_TOML.read_text(encoding="utf-8")
    updated_toml = replace_package_version(cargo_toml, "[package]", "[", "kdeocr", version)
    CARGO_TOML.write_text(updated_toml, encoding="utf-8")


def main() -> None:
    arguments = sys.argv[1:]
    if arguments == ["show"]:
        version = read_version()
        print(f"v{version}")
        return
    if len(arguments) != 2 or arguments[0] != "change":
        raise ValueError("usage: version.py show | version.py change <vMAJOR.MINOR.PATCH>")

    tag = arguments[1]
    if not tag.startswith("v"):
        raise ValueError(f"version must start with v: {tag}")
    version = tag[1:]
    if len(version) > 128 or not SEMVER.fullmatch(version):
        raise ValueError(f"invalid semantic version: {tag}")

    previous = read_version()
    if version == previous:
        print(f"v{previous}")
        return
    write_version(version)
    print(f"v{previous} -> v{version}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"version: {error}", file=sys.stderr)
        raise SystemExit(1) from error
