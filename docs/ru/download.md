# Скачать

<div class="grid cards" markdown>

-   :material-microsoft-windows: **Windows**

    ---

    Установщик (`.exe`), собранный с помощью Inno Setup. Включает модель по умолчанию.

    [:material-download: Скачать установщик Windows](https://github.com/seechov/synth1-patch-generator/releases/latest){ .md-button }

-   :material-apple: **macOS**

    ---

    `Seechov Forge.app` в `.dmg`. Модель размещается в `Contents/MacOS/model/`.

    [:material-download: Скачать macOS DMG](https://github.com/seechov/synth1-patch-generator/releases/latest){ .md-button }

-   :material-linux: **Linux**

    ---

    Пакеты `cargo-deb` и `cargo-generate-rpm`. Модель устанавливается в `/usr/share/seechov-forge/model`.

    [:material-download: Скачать .deb / .rpm](https://github.com/seechov/synth1-patch-generator/releases/latest){ .md-button }

</div>

Все установщики включают модель по умолчанию (`generator.onnx` + `normalization.json`), поэтому приложение загружает её автоматически при первом запуске.

На странице последнего релиза перечислены все артефакты, включая исходный архив и автоматически сгенерированные примечания к релизу.

!!! warning "Windows SmartScreen"
    Установщик Windows **не подписан цифровой подписью**, поэтому Microsoft Edge
    и Windows могут показать предупреждение «Система Windows защитила ваш
    компьютер». Это нормально для неподписанных open-source бинарников. Чтобы
    установить всё равно:

    1. Нажмите **Подробнее**.
    2. Нажмите **Выполнить в любом случае**.

---

## Сборка из исходников

Если вы предпочитаете собрать приложение самостоятельно:

```powershell
cd gui
cargo build --release
```

- Windows: `gui\target\release\seechov-forge.exe`
- Linux:   `gui/target/release/seechov-forge`

См. [Сборка GUI](overview.md#42-gui).
