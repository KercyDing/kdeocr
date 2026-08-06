#!/usr/bin/env bash

set -euo pipefail

version=1.28.0
install_dir=/opt/onnxruntime-$version
temporary_dir=

fail() {
    echo "ONNX Runtime installer: $*" >&2
    exit 1
}

require() {
    command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

run_as_root() {
    if (( EUID == 0 )); then
        "$@"
    else
        require sudo
        sudo "$@"
    fi
}

is_arch_based() {
    [[ -r /etc/os-release ]] || return 1
    local ID= ID_LIKE=
    # os-release is the system-defined source of distribution identifiers.
    source /etc/os-release
    [[ $ID == arch || $ID == cachyos || " $ID_LIKE " == *" arch "* ]]
}

verify_install() {
    require ldconfig
    ldconfig -p | grep -Fq 'libonnxruntime.so' \
        || fail "libonnxruntime.so was not registered with ldconfig"
}

install_arch_package() {
    require pacman
    run_as_root pacman -S --needed onnxruntime-cpu
}

install_official_archive() {
    local architecture archive_url

    require curl
    require install
    require ldconfig
    require mktemp
    require tar

    case $(uname -m) in
        x86_64) architecture=x64 ;;
        aarch64 | arm64) architecture=aarch64 ;;
        *) fail "unsupported architecture: $(uname -m)" ;;
    esac

    temporary_dir=$(mktemp -d)
    trap 'rm -rf -- "$temporary_dir"' EXIT
    archive_url="https://github.com/microsoft/onnxruntime/releases/download/v$version/onnxruntime-linux-$architecture-$version.tgz"

    curl -fL --progress-bar "$archive_url" -o "$temporary_dir/onnxruntime.tgz"
    run_as_root install -d "$install_dir"
    run_as_root tar -xzf "$temporary_dir/onnxruntime.tgz" \
        -C "$install_dir" --strip-components=1
    printf '%s\n' "$install_dir/lib" \
        | run_as_root tee /etc/ld.so.conf.d/onnxruntime.conf >/dev/null
    run_as_root ldconfig
}

if (( $# != 0 )); then
    echo "Usage: $0" >&2
    exit 2
fi

require grep
if is_arch_based; then
    install_arch_package
else
    install_official_archive
fi
verify_install
echo "Installed ONNX Runtime"
