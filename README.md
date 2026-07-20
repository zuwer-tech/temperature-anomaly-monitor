# temperature-anomaly-monitor

Прототип системы раннего обнаружения температурных аномалий на условном
радиохимическом участке. Streamlit-дашборд анализирует температурные ряды с
датчиков, выявляет отклонения и формирует журнал тревог для оператора.

Обнаружение устроено в **два слоя**:

1. **Rule-based (понятные правила)** — резкий скачок, потеря сигнала, сильное
   отклонение от нормы, зависание датчика, отклонение от группы датчиков,
   устойчивый перегрев. Каждую такую тревогу можно объяснить оператору.
2. **Isolation Forest (ИИ)** — ловит «нетипичное поведение», которое сложно
   описать простыми правилами. Обучается только на штатном режиме и
   сохраняется в `models/` (см. [docs/MODEL.md](docs/MODEL.md)).

> 📖 Подробности: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (как устроено),
> [docs/DATA.md](docs/DATA.md) (какие данные), [docs/MODEL.md](docs/MODEL.md)
> (правила и модель).

---

## Быстрый старт (для новичков)

Нужен Python 3.10+.

```bash
# 1. создать виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate          # на Windows: .venv\Scripts\activate

# 2. установить зависимости
pip install -r requirements.txt

# 3. запустить дашборд
streamlit run app.py
```

В браузере откроется дашборд с демонстрационными данными. Слева в боковой
панели можно переключиться на «Загрузить свой CSV» и загрузить свой файл.

---

## Как всё работает (поток данных)

```
 CSV (timestamp, sensor_id, temperature)
        │
        ▼
 preprocessing.py        признаки: rolling_mean, z-score, is_stuck,
 preprocess_data()       отклонение от группы, temp_diff …
        │
        ▼
 anomaly_detection.py    1) правила → rule_anomaly
 detect_anomalies()      2) Isolation Forest (модель из models/) → iforest_anomaly
                        3) final_anomaly = правило | ИИ
                        4) журнал тревог
        │
        ▼
 app.py                  Streamlit-дашборд: графики, anomaly score, журнал тревог
```

Модель Isolation Forest берётся из папки `models/`. Для анализа пользовательского
CSV заранее подготовьте оба артефакта (`scaler.joblib` и `iforest.joblib`):

```bash
python preprocessing.py      # synthetic_temperature_data.csv -> preprocessed_temperature_data.csv
python train_model.py        # обучает на scenario=='normal', сохраняет в models/
```

---

## Структура проекта

| Файл | Назначение |
|---|---|
| `app.py` | Streamlit-дашборд (веб-интерфейс). |
| `preprocessing.py` | Предобработка: считает признаки из сырых температур. |
| `anomaly_detection.py` | Правила + Isolation Forest + журнал тревог. Пороги — в `RULE_PARAMS`. |
| `train_model.py` | Обучение Isolation Forest на штатном режиме + сохранение в `models/`. |
| `data_adapters.py` | Приводит реальные данные `Т2.csv` к схеме пайплайна. |
| `Data.py` | Генератор синтетических данных с разметкой сценариев. |
| `models/` | Обученная модель (`scaler.joblib`, `iforest.joblib`). В git не попадает. |
| `tests/` | pytest-тесты (признаки, сценарии, точность, адаптер). |
| `notebooks/` | Colab-ноутбук для обучения модели. |

| Данные | Что это |
|---|---|
| `Т2.csv` | **Реальные** данные одного датчика (`time_s`, `temp_C`). |
| `synthetic_temperature_data.csv` | Синтетика (генерируется `Data.py`), с разметкой `scenario`. |
| `preprocessed_temperature_data.csv` | Синтетика после `preprocessing.py`. |
| `temperature_anomaly_results.csv`, `alarm_log.csv` | Результаты детекции для демо-режима дашборда. |

---

## Запуск по шагам

### Полный пайплайн в консоли

```bash
python Data.py                  # 1. сгенерировать синтетику (synthetic_temperature_data.csv)
python preprocessing.py          # 2. предобработать -> preprocessed_temperature_data.csv
python train_model.py            # 3. обучить модель на normal -> models/
python anomaly_detection.py      # 4. детекция -> temperature_anomaly_results.csv, alarm_log.csv
```

### Реальные данные (Т2.csv)

```bash
python data_adapters.py          # Т2.csv -> real_temperature_data.csv (каноническая схема)
# затем можно загрузить real_temperature_data.csv в дашборд через «Загрузить свой CSV»
```

### Тесты

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
```

### Обучение в Google Colab (без локального Python)

Открыть `notebooks/train_model_colab.ipynb` в [Colab](https://colab.research.google.com)
и запустить по ячейкам. Подробности — в [docs/MODEL.md](docs/MODEL.md).

---

## Свои данные

CSV должен содержать минимум три колонки:

```csv
timestamp,sensor_id,temperature
2026-06-24 10:00:00,T-01,70.5
2026-06-24 10:01:00,T-01,70.8
2026-06-24 10:02:00,T-01,71.1
```

- `timestamp` — дата и время;
- `sensor_id` — идентификатор датчика;
- `temperature` — температура, число.

Загрузите файл в дашборде через «Загрузить свой CSV». Колонка `scenario`
необязательна (без неё данные помечаются как `user_data`).

---

## Типичные проблемы (troubleshooting)

- **«Обученная модель не найдена или комплект артефактов неполный»** — в
  `models/` нет `scaler.joblib` и/или `iforest.joblib`. Запустите
  `python preprocessing.py`, затем `python train_model.py`. До обучения анализ
  пользовательского CSV не запускается, чтобы исключить data leakage.
- **Реальные данные `Т2.csv` не грузятся в приложение** — у них другая схема
  (`time_s`, `temp_C`, один датчик). Сначала `python data_adapters.py`.
- **`pytest` не запускается** — установите `requirements-dev.txt`.
- **На реальных квантованных данных слишком много «зависаний»** — см. issue по
  объединению `mod_AI_2` (правило зависания настроено под точное равенство).

---

## Лицензия

См. репозиторий.
