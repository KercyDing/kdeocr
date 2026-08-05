#!/usr/bin/env bash

set -euo pipefail

repo=KercyDing/kdeocr
install_dir=${KOCR_INSTALL_DIR:-"$HOME/.local/bin"}
unit_dir=${XDG_CONFIG_HOME:-"$HOME/.config"}/systemd/user
binary_path=$install_dir/kocr
unit_path=$unit_dir/kocr.service
temporary_dir=

fail() {
    echo "kocr installer: $*" >&2
    exit 1
}

require() {
    command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

select_model() {
    local choice=${KOCR_MODEL:-}

    if [[ -z $choice ]]; then
        echo "Select OCR model:" >&2
        echo "  1) ppocrv6-small-r1   24.8 MiB  Recommended" >&2
        echo "  2) ppocrv6-medium-r1  94.9 MiB" >&2
        echo "  3) ppocrv6-tiny-r1     5.4 MiB" >&2
        if [[ -t 0 ]]; then
            read -r -p "Model [1]: " choice
        fi
    fi

    case ${choice:-1} in
        1 | small | ppocrv6-small-r1) printf '%s\n' ppocrv6-small-r1 ;;
        2 | medium | ppocrv6-medium-r1) printf '%s\n' ppocrv6-medium-r1 ;;
        3 | tiny | ppocrv6-tiny-r1) printf '%s\n' ppocrv6-tiny-r1 ;;
        *) fail "unknown model: $choice" ;;
    esac
}

select_action() {
    local choice=

    echo "Select action:" >&2
    echo "  1) Install" >&2
    echo "  2) Uninstall" >&2
    if [[ -t 0 ]]; then
        read -r -p "Action [1]: " choice
    fi

    case ${choice:-1} in
        1 | install) printf '%s\n' install ;;
        2 | uninstall) printf '%s\n' uninstall ;;
        *) fail "unknown action: $choice" ;;
    esac
}

download_binary() {
    local machine target latest_url tag version package archive checksum expected actual release_url

    require curl
    require sha256sum
    require tar

    machine=$(uname -m)
    case "$machine" in
        x86_64) target=x86_64-unknown-linux-gnu ;;
        aarch64 | arm64) target=aarch64-unknown-linux-gnu ;;
        *) fail "unsupported architecture: $machine" ;;
    esac

    if [[ -n ${KOCR_VERSION:-} ]]; then
        tag=v${KOCR_VERSION#v}
    else
        latest_url=$(curl -fsSL -o /dev/null -w '%{url_effective}' \
            "https://github.com/$repo/releases/latest")
        tag=${latest_url##*/}
    fi
    [[ $tag =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$ ]] \
        || fail "invalid release tag: $tag"

    version=${tag#v}
    package=kocr-$version-$target
    archive=$temporary_dir/$package.tar.gz
    checksum=$archive.sha256
    release_url=https://github.com/$repo/releases/download/$tag

    curl -fL --progress-bar "$release_url/$package.tar.gz" -o "$archive"
    curl -fsSL "$release_url/$package.tar.gz.sha256" -o "$checksum"

    expected=$(awk 'NR == 1 { print $1 }' "$checksum")
    actual=$(sha256sum "$archive" | awk '{ print $1 }')
    [[ $expected =~ ^[0-9a-fA-F]{64}$ ]] || fail "release checksum is invalid"
    [[ $actual == "$expected" ]] || fail "release checksum does not match"

    tar -xzf "$archive" -C "$temporary_dir" "$package/kocr"
    printf '%s\n' "$temporary_dir/$package/kocr"
}

install_kocr() {
    local model model_dir source_binary unit_temporary

    require install
    require curl
    require mktemp
    require realpath
    require systemctl
    systemctl --user show-environment >/dev/null \
        || fail "systemd user session is unavailable"

    temporary_dir=$(mktemp -d)
    trap 'rm -rf -- "$temporary_dir"' EXIT

    if [[ -n ${KOCR_BINARY:-} ]]; then
        [[ -x $KOCR_BINARY ]] || fail "binary is not executable: $KOCR_BINARY"
        source_binary=$(realpath -- "$KOCR_BINARY")
    else
        source_binary=$(download_binary)
    fi
    model=$(select_model)

    [[ $binary_path != *$'\n'* && $binary_path != *'"'* ]] \
        || fail "install path contains unsupported characters"
    mkdir -p -- "$install_dir" "$unit_dir"
    install -m 0755 "$source_binary" "$temporary_dir/kocr"
    mv -f -- "$temporary_dir/kocr" "$binary_path"
    model_dir=${XDG_DATA_HOME:-"$HOME/.local/share"}/kdeocr/models/$model
    if [[ ! -f $model_dir/manifest.toml ]]; then
        "$binary_path" install "$model"
    fi
    "$binary_path" use "$model"

    unit_temporary=$(mktemp --tmpdir="$unit_dir" .kocr.service.XXXXXX)
    cat > "$unit_temporary" <<EOF
[Unit]
Description=KOCR global shortcut daemon
PartOf=graphical-session.target
After=graphical-session.target

[Service]
Type=dbus
BusName=io.github.KercyDing.kocr
ExecStart="$binary_path" daemon
Restart=on-failure
RestartSec=2

[Install]
WantedBy=graphical-session.target
EOF
    chmod 0644 "$unit_temporary"
    mv -f -- "$unit_temporary" "$unit_path"

    rm -f -- "${XDG_DATA_HOME:-"$HOME/.local/share"}/applications/kocr-capture.desktop"
    systemctl --user daemon-reload
    systemctl --user enable kocr.service
    systemctl --user restart kocr.service

    echo "Installed kocr to $binary_path"
    echo "Model: $model"
    echo "Shortcut: Alt+1"
}

uninstall_kocr() {
    if [[ -f $unit_path ]]; then
        systemctl --user disable --now kocr.service
        rm -f -- "$unit_path"
        systemctl --user daemon-reload
    fi

    if command -v busctl >/dev/null 2>&1; then
        busctl --user call \
            org.kde.kglobalaccel \
            /kglobalaccel \
            org.kde.KGlobalAccel \
            unregister \
            ss \
            kocr \
            capture-ocr \
            >/dev/null 2>&1 || true
    fi

    rm -f -- "$binary_path"
    rm -f -- "${XDG_DATA_HOME:-"$HOME/.local/share"}/applications/kocr-capture.desktop"
    echo "Uninstalled kocr"
}

usage() {
    echo "Usage: $0" >&2
}

if (( $# != 0 )); then
    usage
    exit 2
fi

case $(select_action) in
    install)
        install_kocr
        ;;
    uninstall)
        uninstall_kocr
        ;;
    *)
        fail "unknown action"
        ;;
esac
