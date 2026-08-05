#!/usr/bin/env bash

set -euo pipefail

if (( $# != 0 )); then
    echo "Usage: $0" >&2
    exit 2
fi

install_dir=${KOCR_INSTALL_DIR:-"$HOME/.local/bin"}
unit_dir=${XDG_CONFIG_HOME:-"$HOME/.config"}/systemd/user
unit_path=$unit_dir/kocr.service

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

rm -f -- "$install_dir/kocr"
rm -f -- "${XDG_DATA_HOME:-"$HOME/.local/share"}/applications/kocr-capture.desktop"
echo "Uninstalled kocr"
