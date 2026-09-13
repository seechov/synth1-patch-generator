# Download

<div class="grid cards" markdown>

-   :material-microsoft-windows: **Windows**

    ---

    Installer (`.exe`) built with Inno Setup. Bundles the default model.

    [:material-download: Download Windows installer](https://github.com/seechov/synth1-patch-generator/releases/latest){ .md-button }

-   :material-apple: **macOS**

    ---

    `Synth1GAN.app` in a `.dmg`. The model ships inside `Contents/MacOS/model/`.

    [:material-download: Download macOS DMG](https://github.com/seechov/synth1-patch-generator/releases/latest){ .md-button }

-   :material-linux: **Linux**

    ---

    `cargo-deb` and `cargo-generate-rpm` packages. The model is installed to `/usr/share/synth1gan/model`.

    [:material-download: Download .deb / .rpm](https://github.com/seechov/synth1-patch-generator/releases/latest){ .md-button }

</div>

All installers bundle the default model (`generator.onnx` + `normalization.json`), so the app loads it automatically on first launch.

The latest release page lists every artifact, including a source bundle and machine-generated release notes.

---

## Build from source

If you'd rather build it yourself:

```powershell
cd gui
cargo build --release
```

- Windows: `gui\target\release\synth1gan.exe`
- Linux:   `gui/target/release/synth1gan`

See [Building the GUI](overview.md#42-building-the-gui) for details.
