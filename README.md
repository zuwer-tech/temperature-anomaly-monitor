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
> (правила и модель), [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)
> (что уже сделано и что осталось).

Перед результатами дашборд показывает активный режим **rules-only** или
**rules+ML** и «паспорт входной пробы»: число строк и датчиков, пропуски,
дубли и проблемы порядка времени. Отдельная таблица группирует соседние
аномальные точки в операторские события. Это учебный прототип, а не
сертифицированная система промышленной безопасности.

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
        ├── data_quality.py        read-only сводка качества
        ▼
 preprocessing.py                 признаки и физическая скорость °C/мин
        │
        ├── rule_config.py         единые пороги инженерных правил
        ├── model_schema.py        единый порядок ML-признаков
        ▼
 anomaly_detection.py             rules-only или rules+ML без обучения на входном CSV
        │
        ├── events.py              группировка точек в события
        ├── evaluation.py          независимые метрики и задержка обнаружения
        ▼
 app.py                           Streamlit: режим, качество, графики, события, журнал
```

Модель Isolation Forest берётся из папки `models/`. Для анализа пользовательского
CSV заранее подготовьте полный комплект (`scaler.joblib`, `iforest.joblib` и
`model_meta.json`):

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
| `anomaly_detection.py` | Правила + заранее обученный Isolation Forest + журнал тревог. |
| `rule_config.py` | Единая конфигурация порогов правил. |
| `model_schema.py` | Канонический состав и порядок ML-признаков. |
| `data_quality.py` | Read-only сводка качества входного CSV. |
| `events.py` | Группировка соседних аномальных точек в события. |
| `evaluation.py` | Независимая оценка rules, ML и combined без `fit`. |
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
python anomaly_detection.py      # 4. детекция -> результаты, тревоги и события
python evaluation.py --output evaluation_report.json  # 5. независимый отчёт качества
```

### Реальные данные (Т2.csv)

```bash
python data_adapters.py          # Т2.csv -> real_temperature_data.csv (каноническая схема)
# затем можно загрузить real_temperature_data.csv в дашборд через «Загрузить свой CSV»
```

### Тесты

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
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
  `models/` нет `scaler.joblib`, `iforest.joblib` и/или `model_meta.json`.
  Для режима **rules+ML** запустите `python preprocessing.py`, затем
  `python train_model.py`. Либо выберите **rules-only**: он применит только
  инженерные правила. В обоих режимах входной CSV не используется для
  обучения, поэтому скрытого data leakage нет.
- **«Сохранённая модель повреждена или несовместима»** — metadata не читается,
  признаки или их порядок изменились либо scaler/model не соответствуют
  текущему коду. Удалять отдельные файлы недостаточно: заново запустите
  `python preprocessing.py`, затем `python train_model.py`. До обучения анализ
  не выполняется.
- **Реальные данные `Т2.csv` не грузятся в приложение** — у них другая схема
  (`time_s`, `temp_C`, один датчик). Сначала `python data_adapters.py`.
- **`pytest` не запускается** — установите `requirements-dev.txt`.
- **На реальных квантованных данных слишком много «зависаний»** — см. issue по
  объединению `mod_AI_2` (правило зависания настроено под точное равенство).

---

## Лицензия

См. репозиторий.
