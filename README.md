# KDEOCR

A small screenshot tool for KDE Plasma 6 on Wayland.

It currently captures a selected region and copies the PNG to the clipboard.

Capture requires `Spectacle` and `wl-copy`. Model installation requires `curl`.

## Capture

```text
kocr capture
kocr capture --ocr
```

## OCR

Recognize text from an image:

```text
kocr image image.png
```

## Models

List and install bundled model profiles:

```text
kocr list
kocr use 1
kocr install ppocrv6-small-r1
kocr install 1
kocr uninstall ppocrv6-small-r1
kocr config
```

`kocr config` opens `~/.config/kdeocr/config.toml` in the editor selected by
`VISUAL`, `EDITOR`, or the desktop default editor.

The file records the selected model and installed model paths:

```toml
model = "ppocrv6-small-r1"

[models.ppocrv6-small-r1]
path = "/home/user/.local/share/kdeocr/models/ppocrv6-small-r1"
```

## License

[Apache License 2.0](LICENSE)
