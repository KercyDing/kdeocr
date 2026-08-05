# KOCR

An offline OCR tool for KDE Plasma 6 on Wayland.

KOCR can recognize an existing PNG or capture a screen region with Spectacle.
Captured images or recognized text can be copied to the Wayland clipboard.

## Install

```text
bash -c "$(curl -fsSL https://raw.githubusercontent.com/KercyDing/kdeocr/main/scripts/install.sh)"
```

Uninstall:

```text
bash -c "$(curl -fsSL https://raw.githubusercontent.com/KercyDing/kdeocr/main/scripts/uninstall.sh)"
```

## Requirements

- Linux with KDE Plasma 6 on Wayland
- `Spectacle` and `wl-copy`
- ONNX Runtime
- `curl` for model installation

## Capture

```text
kocr capture
kocr capture -o
```

The default command copies the captured PNG. `-o` runs OCR, prints the text,
and copies it instead.

## Shortcut

The installer starts a systemd user service and registers `Alt+1` as the
default global shortcut for capture and OCR.

Change `shortcut` with `kocr config`. The running daemon reloads the file after it is saved. Key names follow niri syntax, such as `Alt+1`, `Mod+Shift+Slash`.

## Image

Recognize an existing PNG and print the text:

```text
kocr image test.png
```

## Models

Manage OCR models by ID or name:

```text
kocr list
kocr install 1
kocr use 1
kocr uninstall 1
kocr config
```

`kocr config` opens `~/.config/kdeocr/config.toml` in the editor selected by
`VISUAL`, `EDITOR`, or the desktop default editor.

The file records the selected model and installed model paths:

```toml
shortcut = "Alt+1"
model = "ppocrv6-small-r1"

[models.ppocrv6-small-r1]
path = "/home/user/.local/share/kdeocr/models/ppocrv6-small-r1"
```

## License

[Apache License 2.0](LICENSE)
