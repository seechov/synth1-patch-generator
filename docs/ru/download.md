# Скачать

<div class="grid cards" markdown>

-   :material-microsoft-windows: **Windows**

    ---

    Установщик (`.exe`), собранный с помощью Inno Setup. Включает модель по умолчанию.

    [:material-download: Скачать установщик Windows](https://github.com/seechov/synth1-patch-generator/releases/latest){ .md-button }

-   :material-apple: **macOS**

    ---

    `Synth1GAN.app` в `.dmg`. Модель размещается в `Contents/MacOS/model/`.

    [:material-download: Скачать macOS DMG](https://github.com/seechov/synth1-patch-generator/releases/latest){ .md-button }

-   :material-linux: **Linux**

    ---

    Пакеты `cargo-deb` и `cargo-generate-rpm`. Модель устанавливается в `/usr/share/synth1gan/model`.

    [:material-download: Скачать .deb / .rpm](https://github.com/seechov/synth1-patch-generator/releases/latest){ .md-button }

</div>

Все установщики включают модель по умолчанию (`generator.onnx` + `normalization.json`), поэтому приложение загружает её автоматически при первом запуске.

На странице последнего релиза перечислены все артефакты, включая исходный архив и автоматически сгенерированные примечания к релизу.

---

## Сборка из исходников

Если вы предпочитаете собрать приложение самостоятельно:

```powershell
cd gui
cargo build --release
```

- Windows: `gui\target\release\synth1gan.exe`
- Linux:   `gui/target/release/synth1gan`

См. [Сборка GUI](overview.md#42-gui).
