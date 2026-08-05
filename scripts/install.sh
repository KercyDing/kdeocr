#!/usr/bin/env bash

set -euo pipefail

repo=KercyDing/kdeocr
install_dir=${KOCR_INSTALL_DIR:-"$HOME/.local/bin"}
unit_dir=${XDG_CONFIG_HOME:-"$HOME/.config"}/systemd/user
binary_path=$install_dir/kocr
unit_path=$unit_dir/kocr.service

fail() {
    echo "kocr installer: $*" >&2
    exit 1
}

require() {
    command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
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

if (( $# != 0 )); then
    echo "Usage: $0" >&2
    exit 2
fi

require install
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

[[ $binary_path != *$'\n'* && $binary_path != *'"'* ]] \
    || fail "install path contains unsupported characters"
mkdir -p -- "$install_dir" "$unit_dir"
install -m 0755 "$source_binary" "$temporary_dir/kocr"
mv -f -- "$temporary_dir/kocr" "$binary_path"

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
echo "Shortcut: Alt+1"
