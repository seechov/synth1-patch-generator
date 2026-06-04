# Разбор скрипта `train.py` — Synth1GAN

> Этот документ объясняет работу скрипта построчно.  
> Он рассчитан на читателя, который знаком с программированием на Python, но никогда не занимался обучением нейронных сетей.  
> В каждом разделе — краткое объяснение концепции и ссылки для углублённого изучения.

---

## Содержание

1. [Что делает этот скрипт в целом](#1-что-делает-этот-скрипт-в-целом)
2. [Shebang и docstring — строки 1–15](#2-shebang-и-docstring--строки-1-15)
3. [Импорты — строки 17–32](#3-импорты--строки-17-32)
4. [Словарь параметров Synth1 — строки 36–112](#4-словарь-параметров-synth1--строки-36-112)
5. [Список исключённых колонок TO_DROP — строки 115–131](#5-список-исключённых-колонок-to_drop--строки-115-131)
6. [Категориальные переменные CATEGORICAL_VARS — строки 133–145](#6-категориальные-переменные-categorical_vars--строки-133-145)
7. [Чтение файлов пресетов — строки 151–186](#7-чтение-файлов-пресетов--строки-151-186)
8. [Построение датафрейма — строки 192–199](#8-построение-датафрейма--строки-192-199)
9. [Инженерия признаков — строки 202–251](#9-инженерия-признаков--строки-202-251)
10. [Нормализация данных — строки 254–260](#10-нормализация-данных--строки-254-260)
11. [Архитектура нейросети — строки 266–315](#11-архитектура-нейросети--строки-266-315)
12. [Штраф на градиент — строки 321–340](#12-штраф-на-градиент--строки-321-340)
13. [Цикл обучения — строки 343–432](#13-цикл-обучения--строки-343-432)
14. [Экспорт в ONNX — строки 438–454](#14-экспорт-в-onnx--строки-438-454)
15. [Сохранение нормализации — строки 460–515](#15-сохранение-нормализации--строки-460-515)
16. [Точка входа main() — строки 521–602](#16-точка-входа-main--строки-521-602)
17. [Итоговая схема потока данных](#17-итоговая-схема-потока-данных)

---

## 1. Что делает этот скрипт в целом

Скрипт обучает **GAN** (Generative Adversarial Network, генеративно-состязательную сеть) на коллекции пресетов синтезатора Synth1 в формате `.sy1`. После обучения сеть умеет **генерировать новые правдоподобные пресеты** — настройки синтезатора, которых никогда не существовало, но которые звучат «в стиле» обучающей выборки.

**GAN — что это?**  
GAN — это два нейрона в «состязании»:
- **Генератор** (Generator) — художник, который рисует поддельные картины (пресеты).
- **Дискриминатор** (Discriminator) — детектив, который решает, настоящий ли перед ним пресет или подделка.

Оба тренируются одновременно и улучшают друг друга. В итоге генератор учится делать пресеты настолько хорошо, что дискриминатор не может отличить их от реальных.

**Ссылки для изучения:**
- [Оригинальная статья о GAN (Goodfellow, 2014)](https://arxiv.org/abs/1406.2661)
- [Объяснение GAN простыми словами — Towards Data Science](https://towardsdatascience.com/understanding-generative-adversarial-networks-gans-cd6e4651a29)

---

## 2. Shebang и docstring — строки 1–15

```python
#!/usr/bin/env python3
"""
Synth1GAN — WGAN-GP trainer for Synth1 VST presets.

Usage:
    python train.py --presets-dir C:/path/to/presets --output-dir ./model
    python train.py --presets-dir ./synth_patches/all --output-dir ./model --epochs 5000
...
"""
```

**Строка 1 — shebang** (`#!/usr/bin/env python3`):  
Это директива Unix-систем. Она говорит оболочке, какой интерпретатор использовать для запуска файла, если его запускают напрямую (`./train.py`), а не через `python train.py`. На Windows эта строка игнорируется, но её принято оставлять для переносимости.

**Строки 2–15 — docstring модуля**:  
Строка в тройных кавычках сразу после шебанга — это документационная строка (docstring) всего модуля. Python хранит её в атрибуте `__doc__` и показывает при вызове `help(train)`. Здесь перечислены: что делает скрипт, как его запускать, и каков порядок работы (5 шагов).

---

## 3. Импорты — строки 17–32

```python
import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
```

Импорты делятся на две группы:

### Стандартная библиотека Python (первый блок)

| Библиотека | Зачем |
|---|---|
| `argparse` | Разбирает аргументы командной строки (`--presets-dir`, `--epochs` и т.д.) |
| `csv` | Импортирован, но не используется напрямую (наследие предыдущей версии) |
| `glob` | Ищет файлы по паттерну (`*.sy1`) рекурсивно |
| `json` | Читает и пишет JSON-файлы (сохранение параметров нормализации) |
| `os` | Работа с файловой системой: создание папок, пути |
| `sys` | Доступ к `sys.exit()` для остановки при ошибках, `sys.stdout.flush()` для сброса буфера вывода |
| `pathlib.Path` | Объектно-ориентированная работа с путями (импортирован, используется косвенно) |

### Сторонние библиотеки (второй блок)

**`numpy` (np)**  
Фундаментальная библиотека для численных вычислений в Python. Работает с массивами чисел намного быстрее, чем стандартные списки Python, благодаря тому что хранит данные в памяти непрерывно и вызывает оптимизированный C-код.  
→ [NumPy: официальная документация](https://numpy.org/doc/stable/user/absolute_beginners.html)

**`pandas` (pd)**  
Библиотека для работы с табличными данными — аналог Excel в Python. Основной объект — `DataFrame` (таблица со строками и колонками). Используется здесь для загрузки, очистки и преобразования данных пресетов.  
→ [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)

**`torch` (PyTorch)**  
Главный фреймворк машинного обучения в этом скрипте. PyTorch — это библиотека для создания и обучения нейронных сетей. Её ключевая особенность — **автоматическое дифференцирование** (autograd): PyTorch запоминает все операции над тензорами и умеет автоматически вычислять градиенты, нужные для обучения.  
→ [PyTorch — Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

**`torch.autograd`**  
Модуль автоматического дифференцирования. Используется для ручного вычисления градиентов в функции штрафа (gradient penalty). Без autograd пришлось бы аналитически считать производные — сложно и ненадёжно.

**`torch.nn`**  
Высокоуровневый модуль для построения нейронных сетей. Содержит готовые слои: `Linear` (линейный/полносвязный слой), `BatchNorm1d` (нормализация), `LeakyReLU` (функция активации), `Sequential` (контейнер для последовательных слоёв).

**`sklearn.preprocessing.OneHotEncoder`**  
Из библиотеки scikit-learn. Преобразует категориальные переменные (например, тип формы волны осциллятора: синус, пила, меандр) в числовые векторы. Подробнее — в разделе 9.

**`torch.utils.data.DataLoader, Dataset`**  
Инфраструктура для подачи данных в нейросеть батчами (порциями). `Dataset` — описывает, как получить один пример. `DataLoader` — итерирует по датасету, перемешивает данные и собирает батчи.

---

## 4. Словарь параметров Synth1 — строки 36–112

```python
COL_RENAME = {
    0: "osc1 shape",
    1: "osc2 shape",
    2: "osc2 pitch",
    ...
    74: "portament auto mode",
}
```

Файл `.sy1` хранит параметры синтезатора как числовые идентификаторы: `0, 42, 12` и т.д. Это неудобно для работы. Словарь `COL_RENAME` — это **таблица перевода**: ключ — числовой ID параметра в файле пресета, значение — человекочитаемое название.

Когда DataFrame строится из сырых данных, колонки называются `"0"`, `"1"`, `"2"`. После `df.rename(columns=col_rename_str)` они становятся `"osc1 shape"`, `"osc2 shape"` и т.д.

Всего в Synth1 75 параметров (0–74). Этот словарь покрывает их все.

---

## 5. Список исключённых колонок TO_DROP — строки 115–131

```python
TO_DROP = [
    "name",
    "osc1 fm modulation",
    "osc2 pitch",
    "osc2 kbd track",
    "osc key shift",
    "arp. on/off",
    "arp type",
    "arp range",
    "arp beat",
    "arp gate",
    "equalizer tone",
    "equalizer freq.",
    "equalizer level",
    "equalizer Q",
    "pitch bend range",
]
```

Не все параметры синтезатора одинаково полезны для обучения. Эти колонки исключаются по следующим причинам:

- **`name`** — строка с именем пресета, не является числовым параметром синтеза.
- **Арпеджиатор (`arp.*`)** — часто не заполнен (большинство пресетов не используют арпеджиатор). Разреженные данные с большим количеством нулей мешают обучению.
- **Эквалайзер (`equalizer.*`)** — аналогично, редко используется.
- **`pitch bend range`** — почти всегда одно и то же значение (2 полутона), т.е. не несёт информации о характере пресета.
- **`osc2 pitch`, `osc2 kbd track`, `osc key shift`** — параметры с малой дисперсией или сложной взаимозависимостью.

**Зачем убирать такие колонки?**  
В машинном обучении это называется **feature selection** (отбор признаков). Лишние признаки не просто бесполезны — они добавляют шум и увеличивают пространство поиска, что замедляет обучение и снижает качество модели.  
→ [Feature Selection — Towards Data Science](https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e)

---

## 6. Категориальные переменные CATEGORICAL_VARS — строки 133–145

```python
CATEGORICAL_VARS = [
    "osc1 shape",
    "osc2 shape",
    "osc mod env dest.",
    "filter type",
    "chorus type",
    "play mode type",
    "lfo1 destination",
    "lfo1 type",
    "lfo2 destination",
    "lfo2 type",
]
```

В Synth1 некоторые параметры — не числа в обычном смысле, а **перечисления** (enum). Например, тип формы волны осциллятора: 0 = синусоида, 1 = треугольник, 2 = пила, 3 = меандр. Числа 0, 1, 2, 3 здесь — это **метки**, а не величины. Нельзя сказать, что «пила в два раза больше треугольника».

Для нейросети важно обработать такие параметры иначе, чем обычные числа (громкость, скорость атаки и т.д.). Для этого применяется **one-hot encoding** — подробнее в разделе 9.

---

## 7. Чтение файлов пресетов — строки 151–186

### Функция `read_sy1_file` (строки 151–173)

```python
def read_sy1_file(filepath: str) -> dict | None:
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            lines = f.read().split("\n")
        if len(lines) < 4:
            return None
        header = lines[:3]
        preset: dict = {}
        preset["name"] = header[0].strip()
        preset["color"] = (
            header[1].split("=", 1)[1].strip() if "=" in header[1] else "red"
        )
        preset["ver"] = (
            header[2].split("=", 1)[1].strip() if "=" in header[2] else "106"
        )
        for line in lines[3:]:
            line = line.strip()
            if "," in line:
                param_id, value = line.split(",", 1)
                preset[param_id.strip()] = value.strip()
        return preset
    except Exception:
        return None
```

Формат `.sy1` — это текстовый файл. Первые 3 строки — заголовок:
```
Bass Wobble
color=blue
ver=106
```
Затем идут строки вида `0,127` (параметр ID, значение).

**Разбор функции:**

- `open(filepath, encoding="utf-8", errors="ignore")` — открывает файл. Флаг `errors="ignore"` означает: если встретится байт, который нельзя декодировать как UTF-8, просто пропустить его. Это защищает от краша на файлах со странными кодировками.
- `f.read().split("\n")` — читает весь файл как одну строку, затем разбивает по символу новой строки.
- `if len(lines) < 4: return None` — проверка корректности: если строк меньше 4 (3 заголовка + хотя бы один параметр), файл невалиден.
- `header[0].strip()` — `.strip()` убирает пробелы и символы переноса строки в начале и конце.
- `header[1].split("=", 1)[1]` — разбивает строку `color=blue` по первому символу `=`, берёт вторую часть (`blue`). Параметр `1` означает «максимум одно разбиение», чтобы не сломаться на значениях типа `key=a=b`.
- Цикл `for line in lines[3:]` — обходит все строки после заголовка и разбирает пары `param_id,value`.
- `except Exception: return None` — если что-то пошло не так (файл повреждён, неожиданный формат), функция возвращает `None` вместо краша.

### Функция `load_presets` (строки 176–186)

```python
def load_presets(presets_dir: str) -> list[dict]:
    files = glob.glob(os.path.join(presets_dir, "**", "*.sy1"), recursive=True)
    files += glob.glob(os.path.join(presets_dir, "*.sy1"))
    files = list(set(files))
    presets = [r for f in files if (r := read_sy1_file(f)) is not None]
    return presets
```

- `glob.glob(..., recursive=True)` с паттерном `**/*.sy1` — рекурсивный поиск файлов по маске. `**` означает «любой уровень вложенности папок».
- Второй `glob.glob` для корня — на случай, если пресеты лежат прямо в указанной папке, а не в подпапках.
- `list(set(files))` — множество `set` автоматически удаляет дубликаты (один и тот же файл мог попасть из обоих glob-запросов).
- `[r for f in files if (r := read_sy1_file(f)) is not None]` — list comprehension с **оператором моржа** (`:=`, walrus operator, Python 3.8+). Он присваивает результат `read_sy1_file(f)` переменной `r` прямо внутри условия — так не нужно вызывать функцию дважды.

---

## 8. Построение датафрейма — строки 192–199

```python
def build_dataframe(presets: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(presets)
    df = df.dropna(axis=1, thresh=int(len(df) * 0.5))
    col_rename_str = {str(k): v for k, v in COL_RENAME.items()}
    df = df.rename(columns={k: v for k, v in col_rename_str.items() if k in df.columns})
    return df
```

- `pd.DataFrame(presets)` — превращает список словарей в таблицу. Каждый словарь (пресет) становится строкой; ключи словарей — это имена колонок. Если в одних пресетах есть параметр, которого нет в других, pandas ставит `NaN` (Not a Number) — специальный маркер отсутствия значения.

- `df.dropna(axis=1, thresh=int(len(df) * 0.5))` — удаляет **колонки** (`axis=1`), где меньше 50% (`thresh=0.5 * количество строк`) непустых значений. Иначе говоря: если меньше половины пресетов содержат некий параметр — значит, он нерелевантен или необязателен, и его удаляют.

  - `axis=0` означало бы «удалять строки»
  - `thresh=N` означает «оставить колонку, если в ней хотя бы N непустых значений»

- `col_rename_str` — создаёт копию `COL_RENAME`, где ключи преобразованы из `int` в `str`, потому что pandas хранит имена колонок из `.sy1`-файла как строки (`"0"`, `"1"`).

- `df.rename(columns=...)` — переименовывает колонки по словарю, оставляя без изменения те, которых нет в словаре.

**Ссылки:**
- [Pandas DataFrame — официальная документация](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
- [Работа с пропущенными данными в pandas](https://pandas.pydata.org/docs/user_guide/missing_data.html)

---

## 9. Инженерия признаков — строки 202–251

```python
def engineer_features(df):
    ...
```

Это самая сложная часть предобработки. Разберём пошагово.

### Шаг 1: Удаление ненужных колонок (строки 206–210)

```python
drop_existing = [c for c in TO_DROP if c in df.columns]
df = df.drop(columns=drop_existing)
for meta in ["color", "ver", "pack"]:
    if meta in df.columns:
        df = df.drop(columns=[meta])
```

Удаляются колонки из списка `TO_DROP` (объяснён в разделе 5), а также метаданные пресета (`color`, `ver`, `pack`), которые не описывают звук.

### Шаг 2: Преобразование в числа (строки 213–216)

```python
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(axis=1, thresh=int(len(df) * 0.8))
df = df.fillna(df.median(numeric_only=True))
```

- `pd.to_numeric(df[col], errors="coerce")` — пытается конвертировать значение в число. Если не получается (например, колонка содержит текст), ставит `NaN`. Параметр `errors="coerce"` означает «при ошибке — заменить на NaN», в отличие от `errors="raise"` (выбросить исключение).
- После конвертации снова удаляем колонки, где меньше 80% реальных значений.
- `df.fillna(df.median())` — заполняем оставшиеся `NaN` **медианным значением** по каждой колонке. Медиана предпочтительнее среднего (mean), т.к. устойчива к выбросам. Например, если один пресет имеет экстремальную громкость, это не исказит заполнение.

### Шаг 3: One-hot encoding категориальных переменных (строки 219–247)

```python
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
values = df[col].values.reshape(-1, 1)
enc.fit(values)
encoders[col] = enc

encoded = enc.transform(values)
encoded_df = pd.DataFrame(
    encoded,
    columns=[f"{col}-{i}" for i in range(encoded.shape[1])],
    index=df.index,
)
```

**Почему нельзя просто оставить числа для категорий?**

Представим: форма волны OSC1 кодируется как 0=синус, 1=треугольник, 2=пила, 3=меандр.  
Если передать в нейросеть эти числа напрямую, она будет думать, что «меандр» (3) «в три раза больше» синуса (0), или что треугольник «между» синусом и пилой. Это математически некорректно.

**One-hot encoding** заменяет одно число на вектор из нулей с единицей на нужной позиции:

```
синус    → [1, 0, 0, 0]
треугол. → [0, 1, 0, 0]
пила     → [0, 0, 1, 0]
меандр   → [0, 0, 0, 1]
```

Теперь все категории равноудалены — нейросеть не делает ложных арифметических предположений.

- `enc.fit(values)` — кодировщик смотрит на все уникальные значения в колонке и запоминает их.
- `enc.transform(values)` — применяет кодирование.
- `reshape(-1, 1)` — scikit-learn ожидает данные в форме `(N, 1)` (N строк, 1 колонка). `-1` говорит numpy: «вычисли это измерение сам».
- `f"{col}-{i}"` — новые колонки называются, например, `osc1 shape-0`, `osc1 shape-1`, ...
- `encoders[col] = enc` — сохраняем кодировщик, чтобы потом применить его же при генерации пресетов (для декодирования обратно).

**Ссылки:**
- [One-Hot Encoding — объяснение](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)
- [scikit-learn OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)

### Шаг 4: Удаление дубликатов (строка 247)

```python
df = df.drop_duplicates()
```

Одинаковые пресеты в датасете создают bias — нейросеть выучит, что «этот набор параметров встречается чаще», и будет генерировать его чаще. Удаление дубликатов делает распределение более равномерным.

---

## 10. Нормализация данных — строки 254–260

```python
def normalize(df):
    min_vals = df.min()
    max_vals = df.max()
    range_vals = (max_vals - min_vals).replace(0, 1)
    df_norm = 2.0 * ((df - min_vals) / range_vals) - 1.0
    return df_norm, min_vals, max_vals
```

Параметры синтезатора имеют разные диапазоны:
- Громкость усилителя: от 0 до 127
- Скорость атаки: от 0 до 1
- Время задержки: от 0 до 100

Если подавать их в нейросеть «как есть», слои с большими числами будут доминировать. Нейросеть плохо обучается на данных разных масштабов.

**Формула нормализации** приводит все значения в диапазон `[-1, 1]`:

```
x_norm = 2 * (x - min) / (max - min) - 1
```

- При `x = min`: `2 * 0/range - 1 = -1`
- При `x = max`: `2 * 1 - 1 = 1`
- При `x = среднее`: ≈ 0

Диапазон `[-1, 1]` выбран намеренно — он совпадает с диапазоном функции `Tanh`, которой заканчивается генератор (см. раздел 11).

`.replace(0, 1)` — защита от деления на ноль: если у колонки `min == max` (все значения одинаковы), диапазон равен 0. Заменяем его на 1, чтобы избежать `inf`.

Функция возвращает также `min_vals` и `max_vals` — они нужны для **денормализации** при генерации пресетов (обратный перевод из `[-1, 1]` в реальные значения).

**Ссылки:**
- [Зачем нормализовать данные для нейросетей](https://www.jeremyjordan.me/batch-normalization/)
- [Feature Scaling — Wikipedia](https://en.wikipedia.org/wiki/Feature_scaling)

---

## 11. Архитектура нейросети — строки 266–315

### Класс `Generator` (строки 266–287)

```python
class Generator(nn.Module):
    def __init__(self, latent_dim: int, data_size: int):
        super().__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, data_size),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)
```

**Что делает генератор?**

Генератор принимает **вектор шума** `z` (случайные числа) размером `latent_dim` (по умолчанию 10) и превращает его в вектор параметров синтезатора размером `data_size` (~60 чисел после кодирования).

Это работает как «разворачивание» случайных чисел в осмысленные данные — примерно как раскрытие цветка из семени.

**Архитектура** — это 4 блока, каждый из которых расширяет вектор:
```
10 → 128 → 256 → 512 → 1024 → data_size
```

**Что такое `nn.Linear`?**  
Полносвязный (dense) слой. Каждый нейрон входа соединён с каждым нейроном выхода. Математически: `output = W × input + b`, где `W` — матрица весов, `b` — вектор сдвигов. Именно эти числа (`W` и `b`) и обновляются в процессе обучения.

**Что такое `nn.BatchNorm1d`?**  
Batch Normalization — нормализует активации внутри батча. Это стабилизирует обучение: без нормализации значения могут «взрываться» (становиться очень большими) или «угасать» (стремиться к нулю) по мере прохождения через слои.  
→ [Batch Normalization: Accelerating Deep Network Training (оригинальная статья)](https://arxiv.org/abs/1502.03167)

**Что такое `nn.LeakyReLU`?**  
Функция активации. Без нелинейных функций активации несколько линейных слоёв эквивалентны одному — нейросеть не выучила бы ничего сложного.

`LeakyReLU(0.2)` работает так:
```
если x > 0: output = x
если x ≤ 0: output = 0.2 * x
```

Это улучшенная версия обычного ReLU (`max(0, x)`), которая не «убивает» нейроны с отрицательными активациями (проблема «мёртвых нейронов»).  
→ [Функции активации — объяснение](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html)

**`nn.Tanh` в конце:**  
Гиперболический тангенс сжимает любое число в диапазон `(-1, 1)`. Это идеально совпадает с нашей нормализацией данных в диапазон `[-1, 1]` — таким образом, выход генератора напрямую интерпретируется как нормализованные параметры пресета.

**`nn.Sequential`:**  
Контейнер, который применяет слои последовательно. Оператор `*block(...)` «разворачивает» список слоёв в аргументы Sequential.

**`def forward(self, z)`:**  
Это обязательный метод PyTorch-модели. Он описывает прямой проход (forward pass): как данные текут через сеть. При вызове `generator(z)` PyTorch автоматически вызывает `forward`.

---

### Класс `Discriminator` (строки 290–304)

```python
class Discriminator(nn.Module):
    def __init__(self, data_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(data_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
        )
```

Дискриминатор работает в обратном направлении: принимает вектор параметров пресета и «сворачивает» его в одно число:
```
data_size → 256 → 128 → 64 → 1
```

Это число — оценка «реальности» пресета. В стандартном GAN это вероятность [0, 1]. В WGAN-GP (который используется здесь) дискриминатор называется **критиком** (critic) и выдаёт произвольное вещественное число — чем выше, тем «реальнее» пресет.

Обратите внимание: здесь **нет BatchNorm**. Это намеренно — WGAN-GP требует, чтобы у критика не было BatchNorm (это бы нарушило условие Липшица, необходимое для градиентного штрафа).

---

### Класс `PresetDataset` (строки 307–315)

```python
class PresetDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = df.values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.FloatTensor(self.data[idx])
```

Это адаптер между pandas DataFrame и PyTorch. `DataLoader` (который обеспечивает батчевую загрузку) требует объект типа `Dataset` с двумя методами:
- `__len__` — сколько примеров всего.
- `__getitem__` — дай мне пример с индексом `idx`.

- `df.values` — извлекает данные из DataFrame как numpy-массив.
- `.astype(np.float32)` — конвертирует в 32-битные числа с плавающей точкой. Это стандарт для нейросетей: `float32` в два раза меньше по памяти, чем `float64`, и GPU обрабатывает его быстрее.
- `torch.FloatTensor(...)` — оборачивает numpy-массив в PyTorch tensor.

**Ссылки:**
- [Создание датасетов в PyTorch](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

---

## 12. Штраф на градиент — строки 321–340

```python
def compute_gradient_penalty(D, real, fake, device):
    alpha = torch.rand(real.size(0), 1, device=device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = D(interpolates)
    ones = torch.ones(real.size(0), 1, device=device, requires_grad=False)
    grads = autograd.grad(
        outputs=d_interp,
        inputs=interpolates,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(grads.size(0), -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()
```

Это сердце алгоритма **WGAN-GP** (Wasserstein GAN with Gradient Penalty). Понять его сложнее всего, поэтому объясним по шагам.

**Проблема обычных GAN:**  
В классическом GAN дискриминатор может стать «слишком хорошим» и начать выдавать числа `0.0000001` для фейков и `0.9999999` для реальных — градиент становится почти нулевым, и генератор перестаёт обучаться (это называется **mode collapse** или **vanishing gradient**).

**Решение WGAN:**  
Вместо вероятностей дискриминатор (теперь «критик») выдаёт произвольные числа, а функция потерь считает **Wasserstein distance** — «транспортное расстояние» между распределениями реальных и фейковых данных. Это более стабильно.

**Проблема WGAN:**  
Wasserstein distance корректна только если критик удовлетворяет **условию Липшица**: градиент функции критика не должен нигде превышать 1. В оригинальном WGAN это достигалось грубым обрезанием весов — плохой метод.

**Решение WGAN-GP:**  
Вместо обрезания весов добавляется **штраф**: если норма градиента критика на «интерполированных» точках между реальными и фейковыми отклоняется от 1 — это добавляется к функции потерь как штраф.

**Разбор кода:**

1. `alpha = torch.rand(batch_size, 1)` — случайное число от 0 до 1 для каждого элемента батча.

2. `interpolates = alpha * real + (1 - alpha) * fake` — **интерполяция** между реальным и фейковым пресетом. При `alpha=1` получаем реальный пресет, при `alpha=0` — фейковый, при `alpha=0.5` — смесь.

3. `.requires_grad_(True)` — говорим PyTorch отслеживать градиенты для этого тензора (по умолчанию не отслеживается).

4. `d_interp = D(interpolates)` — прогоняем интерполированные примеры через критик.

5. `autograd.grad(...)` — вычисляем градиент выхода критика по входу (интерполированным точкам). Это `∂D/∂interpolates`.
   - `create_graph=True` — нужно, чтобы потом взять градиент от самого штрафа (градиент градиента).
   - `retain_graph=True` — не удалять вычислительный граф после взятия градиента.

6. `grads.view(grads.size(0), -1)` — «расплющиваем» градиент в вектор по каждому примеру батча.

7. `grads.norm(2, dim=1)` — L2-норма (евклидова длина) вектора градиента для каждого примера.

8. `((norm - 1) ** 2).mean()` — штраф: насколько сильно норма отклоняется от 1. В идеале это должно быть 0.

**Ссылки:**
- [Оригинальная статья WGAN-GP](https://arxiv.org/abs/1704.00028)
- [Объяснение Wasserstein GAN простыми словами](https://jonathan-hui.medium.com/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490)

---

## 13. Цикл обучения — строки 343–432

```python
def train(df_norm, output_dir, n_epochs, batch_size, latent_dim, lr, n_critic, lambda_gp, sample_interval):
```

Это главная функция обучения. Разберём её по частям.

### Настройка устройства и директорий (строки 354–365)

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.85)
```

**GPU vs CPU:**  
Нейросети обучаются в 10–100 раз быстрее на GPU (видеокарте) по сравнению с CPU. Это потому что GPU состоит из тысяч маленьких ядер, оптимизированных для параллельных матричных вычислений — именно того, что нужно нейросетям.

`torch.cuda.is_available()` — проверяет, есть ли CUDA-совместимая видеокарта (NVIDIA). Если есть — используем GPU, иначе — CPU.

`set_per_process_memory_fraction(0.85)` — ограничивает потребление видеопамяти до 85%. Это защита от `CUDA out of memory` ошибок на видеокартах с малым объёмом памяти.

### DataLoader (строки 367–375)

```python
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    pin_memory=True,
)
```

- `batch_size=64` — нейросеть обучается не на одном примере за раз и не на всех сразу, а на **батчах** (мини-выборках). Размер 64 — типичный компромисс между скоростью и стабильностью обучения.
- `shuffle=True` — перемешивать данные перед каждой эпохой. Это предотвращает запоминание порядка примеров.
- `drop_last=True` — если последний батч меньше `batch_size` (датасет не делится нацело), его выбрасывают. BatchNorm плохо работает на очень маленьких батчах.
- `num_workers=0` — количество параллельных процессов для загрузки данных. 0 = в основном потоке. На Windows с 0 меньше проблем с multiprocessing.
- `pin_memory=True` — размещает данные в «прижатой» (pinned) памяти CPU, что ускоряет передачу на GPU.

**Что такое эпоха?**  
Одна эпоха = один полный проход по всему датасету. При 1000 пресетах и batch_size=64 в одной эпохе будет `1000 // 64 = 15` батчей (итераций).

### Оптимизаторы (строки 380–381)

```python
opt_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
```

**Оптимизатор** — алгоритм, который обновляет веса нейросети на основе градиентов. **Adam** (Adaptive Moment Estimation) — один из наиболее популярных алгоритмов, адаптирующий learning rate для каждого параметра.

- `lr=0.0002` — **learning rate** (скорость обучения). Это то, на сколько изменяются веса после каждого батча. Слишком большой → нестабильное обучение. Слишком маленький → медленная сходимость.
- `betas=(0.5, 0.999)` — коэффициенты экспоненциального скользящего среднего первого и второго момента. `beta1=0.5` (вместо стандартного 0.9) — стандартная практика для GAN, рекомендованная в статье DCGAN.

**Ссылки:**
- [Adam optimizer — оригинальная статья](https://arxiv.org/abs/1412.6980)
- [Визуализация оптимизаторов](https://www.ruder.io/optimizing-gradient-descent/)

### Основной цикл (строки 389–428)

```python
for epoch in range(n_epochs):
    for i, real_imgs in enumerate(dataloader):
```

Внешний цикл — по эпохам. Внутренний — по батчам внутри эпохи.

#### Шаг дискриминатора / критика

```python
opt_D.zero_grad()
z = torch.randn(real_imgs.size(0), latent_dim, device=device)
fake_imgs = generator(z).detach()

real_val = discriminator(real_imgs)
fake_val = discriminator(fake_imgs)
gp = compute_gradient_penalty(discriminator, real_imgs, fake_imgs, device)
d_loss = -real_val.mean() + fake_val.mean() + lambda_gp * gp
d_loss.backward()
opt_D.step()
```

1. `opt_D.zero_grad()` — обнуляем накопленные градиенты. PyTorch по умолчанию **добавляет** градиенты, а не перезаписывает — если не обнулить, предыдущие батчи повлияют на текущий.

2. `z = torch.randn(...)` — генерируем **вектор шума** из стандартного нормального распределения N(0,1). Каждое число — случайное, из «гауссовского» (колоколообразного) распределения.

3. `fake_imgs = generator(z).detach()` — генерируем фейковые пресеты. `.detach()` отсоединяет результат от вычислительного графа генератора — нам не нужно обновлять генератор в этом шаге.

4. `real_val` и `fake_val` — оценки критика для реальных и фейковых пресетов.

5. **Функция потерь WGAN-GP:**
   ```
   L_D = -E[D(real)] + E[D(fake)] + λ * gradient_penalty
   ```
   Критик хочет максимизировать `D(real) - D(fake)` (разницу оценок реального и фейкового). Функция потерь — это отрицательная такая разница (потому что мы **минимизируем** потерю).

6. `d_loss.backward()` — **обратное распространение ошибки** (backpropagation). PyTorch идёт по вычислительному графу в обратном направлении и вычисляет, как изменение каждого веса влияет на потерю.

7. `opt_D.step()` — применяет обновление весов: `w := w - lr * gradient`.

**Что такое backpropagation?**  
Это алгоритм вычисления градиентов в нейросети. Используется цепное правило дифференцирования: `∂L/∂w = ∂L/∂y * ∂y/∂w`, применённое рекурсивно от выходного слоя к входному.  
→ [Backpropagation — 3Blue1Brown (видео)](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

#### Шаг генератора (строки 406–421)

```python
if i % n_critic == 0:
    opt_G.zero_grad()
    z = torch.randn(real_imgs.size(0), latent_dim, device=device)
    fake_imgs = generator(z)
    g_loss = -discriminator(fake_imgs).mean()
    g_loss.backward()
    opt_G.step()
```

- `if i % n_critic == 0` — генератор обновляется **реже** критика: раз в `n_critic=5` итераций критика. Это важная особенность WGAN: критику нужно больше шагов, чтобы хорошо оценивать. Если обновлять их поровну, критик не успевает «созреть».

- `g_loss = -discriminator(fake_imgs).mean()` — функция потерь генератора: он хочет, чтобы критик **высоко оценивал** его фейки. Генератор обновляется, чтобы максимизировать `D(fake)`, т.е. минимизировать `-D(fake)`.

- На этот раз **нет `.detach()`** — нам нужны градиенты, которые протекут через дискриминатор обратно к генератору.

### Чекпоинты (строки 425–428)

```python
if epoch > 0 and epoch % 500 == 0:
    ckpt = os.path.join(checkpoints_dir, f"generator_{epoch}.pt")
    torch.save(generator.state_dict(), ckpt)
```

Каждые 500 эпох сохраняется **чекпоинт** — снимок весов генератора. Это позволяет:
- Восстановить обучение после прерывания.
- Сравнить качество на разных стадиях обучения.
- Выбрать лучшую версию, если позднее качество ухудшилось (переобучение).

`generator.state_dict()` — возвращает словарь `{имя_параметра: тензор_весов}`. Это стандартный формат сохранения моделей в PyTorch.

---

## 14. Экспорт в ONNX — строки 438–454

```python
def export_onnx(generator, latent_dim, output_dir):
    generator.eval()
    dummy = torch.randn(1, latent_dim).to(next(generator.parameters()).device)
    path = os.path.join(output_dir, "generator.onnx")
    torch.onnx.export(
        generator,
        dummy,
        path,
        input_names=["noise"],
        output_names=["preset"],
        dynamic_axes={"noise": {0: "batch"}, "preset": {0: "batch"}},
        opset_version=13,
        do_constant_folding=True,
    )
```

**Что такое ONNX?**  
ONNX (Open Neural Network Exchange) — это **универсальный формат** для нейросетей, независимый от фреймворка. PyTorch модель можно экспортировать в `.onnx`, а затем загрузить в C++, Rust, Java — где угодно, без зависимости от Python и PyTorch.

GUI-приложение на Rust использует ONNX Runtime для запуска генератора.

**Разбор кода:**

- `generator.eval()` — переключает модель в режим **inference** (инференс/вывод). В этом режиме BatchNorm и Dropout (если бы он был) работают иначе, чем при обучении. Важно не забывать об этом.

- `dummy = torch.randn(1, latent_dim)` — «заглушка» входных данных нужна для трассировки: PyTorch прогоняет через модель реальные данные, записывает все операции — это и становится ONNX-графом.

- `.to(next(generator.parameters()).device)` — убеждаемся, что тензор на том же устройстве (GPU/CPU), что и модель.

- `input_names=["noise"]` — имя входного тензора в ONNX-графе (для удобства в коде GUI).
- `output_names=["preset"]` — имя выходного тензора.
- `dynamic_axes={"noise": {0: "batch"}}` — говорим ONNX, что первое измерение (batch size) может быть произвольным. Без этого модель была бы скомпилирована только для batch_size=1.
- `opset_version=13` — версия набора операций ONNX. Версия 13 поддерживается большинством современных ONNX-рантаймов.
- `do_constant_folding=True` — оптимизация: вычисляет константные подвыражения заранее.

**Ссылки:**
- [ONNX — официальный сайт](https://onnx.ai/)
- [Экспорт PyTorch в ONNX](https://pytorch.org/docs/stable/onnx.html)

---

## 15. Сохранение нормализации — строки 460–515

```python
def save_normalization(df_reduced, min_vals, max_vals, cat_vars, encoders, col_names, latent_dim, output_dir):
```

После обучения генератор выдаёт числа в диапазоне `[-1, 1]`. Чтобы превратить их обратно в реальные параметры синтезатора, GUI должно знать:
- Для каждого параметра: что значит `-1` и что значит `1` (т.е. исходные `min` и `max`).
- Для категориальных параметров: какому числовому ID соответствует каждая категория.

Эта функция сохраняет всё это в `normalization.json`.

```python
continuous_stats[col] = {
    "col_idx": idx,
    "min": float(min_vals[col]),
    "max": float(max_vals[col]),
    "param_id": name_to_id.get(col, col),
}
```

Для каждой непрерывной колонки:
- `col_idx` — позиция в выходном векторе (чтобы GUI знало, какое число генератора соответствует этому параметру).
- `min`, `max` — для денормализации: `x_real = (x_norm + 1) / 2 * (max - min) + min`.
- `param_id` — числовой ID параметра Synth1, который нужно установить.

```python
categorical_encodings[col] = {
    "start_idx": start_idx,
    "num_classes": len(categories),
    "categories": [int(c) for c in categories],
    "param_id": name_to_id.get(col, col),
}
```

Для категориальных параметров:
- `start_idx` — с какой позиции начинается one-hot вектор в выходном тензоре.
- `num_classes` — длина one-hot вектора.
- `categories` — список реальных значений: `[0, 1, 2, 3]` для 4 форм волны.

GUI берёт `argmax` (позицию максимального числа) из one-hot части и смотрит, какому ID это соответствует.

---

## 16. Точка входа main() — строки 521–602

```python
def main() -> None:
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--presets-dir", required=True, ...)
    parser.add_argument("--output-dir", default="./model", ...)
    parser.add_argument("--epochs", type=int, default=20000, ...)
    ...
    args = parser.parse_args()
```

### Аргументы командной строки

`argparse` — стандартный модуль Python для разбора аргументов. Вот параметры и их смысл:

| Аргумент | По умолчанию | Смысл |
|---|---|---|
| `--presets-dir` | обязательный | Папка с `.sy1` файлами |
| `--output-dir` | `./model` | Куда сохранять модель |
| `--epochs` | 20000 | Сколько полных проходов по датасету |
| `--batch-size` | 64 | Размер мини-выборки |
| `--latent-dim` | 10 | Размер вектора шума для генератора |
| `--lr` | 0.0002 | Learning rate (скорость обучения) |
| `--n-critic` | 5 | Сколько шагов критика на 1 шаг генератора |
| `--lambda-gp` | 10.0 | Вес штрафа на градиент |
| `--sample-interval` | 400 | Как часто выводить лог в консоль |

### Порядок выполнения (строки 549–598)

```python
# 1. Load
presets = load_presets(args.presets_dir)
if len(presets) < args.batch_size:
    sys.exit(1)

# 2. Build DataFrame
df = build_dataframe(presets)

# 3. Feature engineering
df_reduced, cat_vars, encoders = engineer_features(df)

# 4. Normalize
df_norm, min_vals, max_vals = normalize(df_reduced)
col_names = list(df_norm.columns)

# 5. Train
generator = train(df_norm, args.output_dir, ...)

# 6. Export
export_onnx(generator, args.latent_dim, args.output_dir)
save_normalization(...)
```

Проверка `if len(presets) < args.batch_size` — важная защита. Если датасет меньше одного батча, нейросеть не может обучиться: DataLoader с `drop_last=True` просто выбросит единственный неполный батч, и цикл обучения ничего не сделает.

### Финальные строки (строки 601–602)

```python
if __name__ == "__main__":
    main()
```

Это стандартная идиома Python: код в этом блоке запускается **только если файл запущен напрямую** (`python train.py`), но не если он импортирован как модуль (`import train`). Это позволяет использовать функции из `train.py` в других скриптах без автоматического запуска обучения.

---

## 17. Итоговая схема потока данных

```
.sy1 файлы
    │
    ▼ read_sy1_file()
список словарей {param_id: value, ...}
    │
    ▼ build_dataframe()
DataFrame с числовыми ID → читаемыми именами
    │
    ▼ engineer_features()
    ├── удаление шумных колонок (TO_DROP)
    ├── конвертация в числа
    ├── one-hot encoding категорий
    └── удаление дубликатов
DataFrame с признаками
    │
    ▼ normalize()
DataFrame в диапазоне [-1, 1]  +  min/max для денормализации
    │
    ▼ train()
    │
    ├── каждый батч:
    │   ├── 5× шаг критика:
    │   │   ├── генерировать шум z → G(z) = fake_preset
    │   │   ├── оценить real: D(real_preset)
    │   │   ├── оценить fake: D(fake_preset)
    │   │   ├── вычислить gradient penalty
    │   │   └── обновить веса D
    │   └── 1× шаг генератора:
    │       ├── генерировать шум z → G(z)
    │       ├── критик оценивает: D(G(z))
    │       └── обновить веса G, чтобы D(G(z)) ↑
    │
    └── каждые 500 эпох: сохранить чекпоинт
    │
    ▼ export_onnx()
generator.onnx  (универсальный формат для GUI на Rust)
    │
    ▼ save_normalization()
normalization.json  (min/max + категории для декодирования)
    │
    ▼
GUI принимает случайный вектор → ONNX Runtime → денормализация → Synth1 пресет
```

---

## Ключевые концепции для дальнейшего изучения

| Концепция | Что это | Ресурс |
|---|---|---|
| Нейронная сеть | Универсальный аппроксиматор функций | [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) |
| Backpropagation | Алгоритм вычисления градиентов | [Backprop — Andrej Karpathy](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b) |
| GAN | Генеративно-состязательные сети | [GAN — Ian Goodfellow (оригинал)](https://arxiv.org/abs/1406.2661) |
| WGAN-GP | Улучшенный GAN с W-расстоянием | [WGAN-GP статья](https://arxiv.org/abs/1704.00028) |
| Batch Normalization | Нормализация активаций | [BN статья](https://arxiv.org/abs/1502.03167) |
| Adam optimizer | Адаптивный оптимизатор | [Adam статья](https://arxiv.org/abs/1412.6980) |
| One-hot encoding | Кодирование категорий | [ML Mastery](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/) |
| ONNX | Формат для переносимых моделей | [onnx.ai](https://onnx.ai/) |
| PyTorch | Фреймворк для нейросетей | [PyTorch Tutorials](https://pytorch.org/tutorials/) |
