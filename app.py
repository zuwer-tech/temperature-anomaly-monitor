import pandas as pd
import streamlit as st
import plotly.express as px

from preprocessing import preprocess_data
from anomaly_detection import (
    ANALYSIS_MODE_RULES_ML,
    ANALYSIS_MODE_RULES_ONLY,
    ModelCompatibilityError,
    ModelNotTrainedError,
    detect_anomalies,
)

# ============================================================
# 1. НАСТРОЙКИ СТРАНИЦЫ
# ============================================================

st.set_page_config(
    page_title="Temperature Anomaly Monitor",
    page_icon="🌡️",
    layout="wide"
)


# ============================================================
# 2. ЗАГРУЗКА ДАННЫХ
# ============================================================

@st.cache_data
def load_demo_data():
    results = pd.read_csv("temperature_anomaly_results.csv")
    alarms = pd.read_csv("alarm_log.csv")

    results["timestamp"] = pd.to_datetime(results["timestamp"])
    alarms["Время"] = pd.to_datetime(alarms["Время"])

    if "analysis_mode" not in results.columns:
        results["analysis_mode"] = ANALYSIS_MODE_RULES_ML
    if "Режим_анализа" not in alarms.columns:
        alarms["Режим_анализа"] = ANALYSIS_MODE_RULES_ML

    return results, alarms


st.sidebar.header("📂 Источник данных")

data_source = st.sidebar.radio(
    "Выберите режим:",
    ["Демонстрационные данные", "Загрузить свой CSV"]
)

if data_source == "Демонстрационные данные":
    df, alarm_log = load_demo_data()
    st.sidebar.info("Используются демонстрационные данные проекта.")

else:
    uploaded_file = st.sidebar.file_uploader(
        "Загрузите CSV-файл",
        type=["csv"]
    )

    if uploaded_file is None:
        st.warning("Загрузите CSV-файл для анализа.")
        st.stop()

    analysis_mode = st.sidebar.radio(
        "Режим анализа:",
        [ANALYSIS_MODE_RULES_ML, ANALYSIS_MODE_RULES_ONLY],
        help=(
            "rules+ML использует инженерные правила и заранее обученную модель; "
            "rules-only — только инженерные правила."
        ),
    )
    use_ml = analysis_mode == ANALYSIS_MODE_RULES_ML

    raw_df = pd.read_csv(uploaded_file)

    try:
        preprocessed_df = preprocess_data(raw_df)
    except TypeError as error:
        st.error(f"Некорректный формат входных данных: {error}")
        st.stop()
    except ValueError as error:
        st.error(f"Некорректные данные в CSV: {error}")
        st.stop()

    try:
        df, alarm_log = detect_anomalies(preprocessed_df, use_ml=use_ml)

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        alarm_log["Время"] = pd.to_datetime(alarm_log["Время"])

        st.sidebar.success("CSV успешно загружен и проанализирован.")

    except ModelNotTrainedError as error:
        st.error(str(error))
        st.info("Подготовьте данные и заранее обучите модель:")
        st.code("python preprocessing.py\npython train_model.py", language="bash")
        st.stop()
    except ModelCompatibilityError as error:
        st.error(str(error))
        st.info("Сохранённая модель повреждена или несовместима. Переобучите её:")
        st.code("python preprocessing.py\npython train_model.py", language="bash")
        st.stop()
    except Exception as error:
        st.error("Во время анализа файла произошла ошибка.")
        st.exception(error)
        st.stop()


# ============================================================
# 3. ЗАГОЛОВОК
# ============================================================

st.title("🌡️ ИИ-модуль мониторинга температурных показаний")
st.markdown(
    """
    Дашборд показывает работу прототипа системы раннего обнаружения температурных аномалий
    на условном радиохимическом участке.
    
    Система анализирует температурные ряды, выявляет отклонения и формирует журнал тревог
    для оператора.
    """
)

active_modes = df["analysis_mode"].dropna().unique().tolist()
if len(active_modes) != 1:
    st.error("Не удалось однозначно определить режим анализа результатов.")
    st.stop()

active_mode = active_modes[0]
if active_mode == ANALYSIS_MODE_RULES_ML:
    st.success(
        "Режим анализа: rules+ML — инженерные правила и Isolation Forest."
    )
elif active_mode == ANALYSIS_MODE_RULES_ONLY:
    st.warning(
        "Режим анализа: rules-only — только инженерные правила; "
        "ИИ-модель не применялась."
    )
else:
    st.error(f"Неизвестный режим анализа: {active_mode}")
    st.stop()


# ============================================================
# 4. БОКОВАЯ ПАНЕЛЬ ФИЛЬТРОВ
# ============================================================

st.sidebar.header("⚙️ Фильтры")

all_sensors = sorted(df["sensor_id"].unique())

selected_sensors = st.sidebar.multiselect(
    "Выберите датчики:",
    options=all_sensors,
    default=all_sensors
)
# Фильтр по сценарию, если такая колонка есть в данных
possible_scenario_columns = ["scenario", "anomaly_type", "scenario_name", "mode", "true_scenario"]

scenario_column = None
for col in possible_scenario_columns:
    if col in df.columns:
        scenario_column = col
        break

# Русские названия сценариев для интерфейса
scenario_names_ru = {
    "normal": "Нормальная работа",
    "sharp_jump": "Резкий скачок температуры",
    "spike": "Резкий выброс",
    "slow_overheating": "Медленный перегрев",
    "sensor_drift": "Дрейф датчика",
    "stuck_sensor": "Зависание датчика",
    "high_noise": "Сильный шум",
    "signal_loss": "Потеря сигнала",
    "correlated_growth": "Коррелированный рост"
}

if scenario_column is not None:
    all_scenarios = sorted(df[scenario_column].dropna().unique())

    selected_scenarios_ru = st.sidebar.multiselect(
        "Сценарий:",
        options=all_scenarios,
        default=all_scenarios,
        format_func=lambda x: scenario_names_ru.get(x, x)
    )

    selected_scenarios = selected_scenarios_ru
else:
    selected_scenarios = None


if len(alarm_log) > 0:
    risk_levels = sorted(alarm_log["Уровень"].unique())
else:
    risk_levels = []

selected_risk_levels = st.sidebar.multiselect(
    "Уровень тревоги:",
    options=risk_levels,
    default=risk_levels
)
show_anomalies_only = st.sidebar.checkbox(
    "Показывать только аномалии",
    value=True
)

# ============================================================
# 5. ФИЛЬТРАЦИЯ ДАННЫХ
# ============================================================

filtered_df = df[df["sensor_id"].isin(selected_sensors)].copy()

# Применяем фильтр сценариев ко всей таблице результатов
if scenario_column is not None and selected_scenarios is not None:
    filtered_df = filtered_df[
        filtered_df[scenario_column].isin(selected_scenarios)
    ].copy()

if len(alarm_log) > 0:
    filtered_alarm_log = alarm_log[
        (alarm_log["Датчик"].isin(selected_sensors)) &
        (alarm_log["Уровень"].isin(selected_risk_levels))
    ].copy()
else:
    filtered_alarm_log = alarm_log.copy()

# Применяем фильтр сценариев к журналу тревог
if "Истинный_сценарий" in filtered_alarm_log.columns and selected_scenarios is not None:
    filtered_alarm_log = filtered_alarm_log[
        filtered_alarm_log["Истинный_сценарий"].isin(selected_scenarios)
    ].copy()
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Обзор",
    "Температурный тренд",
    "Обнаружение аномалий",
    "Журнал аварийных сигналов",
    "О проекте"
])


# ============================================================
# 6. ОСНОВНЫЕ МЕТРИКИ
# ============================================================

with tab1:
    st.subheader("📊 Общий обзор системы")

    # Основные показатели
    total_points = len(filtered_df)
    total_anomalies = int(filtered_df["final_anomaly"].sum())
    anomaly_percent = total_anomalies / total_points * 100 if total_points > 0 else 0

    sensors_count = len(selected_sensors)
    total_alarms = len(filtered_alarm_log)

    high_count = len(filtered_alarm_log[filtered_alarm_log["Уровень"] == "High"])
    medium_count = len(filtered_alarm_log[filtered_alarm_log["Уровень"] == "Medium"])
    warning_count = len(filtered_alarm_log[filtered_alarm_log["Уровень"] == "Warning"])

    # Логика статуса системы
    if high_count > 0:
        system_status = "🔴 Требуется внимание"
        status_text = "Обнаружены тревоги высокого уровня. Необходимо проверить журнал событий."
    elif medium_count > 0 or warning_count > 0:
        system_status = "🟡 Есть предупреждения"
        status_text = "Обнаружены предупреждения или тревоги среднего уровня."
    else:
        system_status = "🟢 Нормальная работа"
        status_text = "Критических тревог по выбранным фильтрам не обнаружено."

    # Карточка статуса
    st.markdown("### Состояние системы")
    st.info(f"**{system_status}**  \n{status_text}")

    # Метрики первого ряда
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Датчиков выбрано", sensors_count)
    col2.metric("Всего точек", total_points)
    col3.metric("Найдено аномалий", total_anomalies)
    col4.metric("Доля аномалий", f"{anomaly_percent:.2f}%")

    # Метрики второго ряда
    col5, col6, col7, col8 = st.columns(4)

    col5.metric("Всего тревог", total_alarms)
    col6.metric("High", high_count)
    col7.metric("Medium", medium_count)
    col8.metric("Warning", warning_count)
    st.markdown("### Выбранные сценарии")

    if scenario_column is not None and len(filtered_df) > 0:
        scenario_counts = (
            filtered_df[scenario_column]
            .value_counts()
            .reset_index()
        )

        scenario_counts.columns = ["Сценарий", "Количество точек"]

        scenario_counts["Сценарий"] = scenario_counts["Сценарий"].map(
            lambda x: scenario_names_ru.get(x, x)
        )

        st.dataframe(
            scenario_counts,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Сценарии не найдены в данных.")
    st.markdown("---")

    st.markdown(
        """
        **Что показывает этот раздел:**  
        Overview дает быстрый обзор состояния системы по выбранным фильтрам.  
        Здесь можно сразу увидеть, сколько данных анализируется, сколько найдено аномалий,
        есть ли тревоги высокого уровня и насколько стабильна работа датчиков.
        """
    )

# ============================================================
# 7. ГРАФИК ТЕМПЕРАТУР
# ============================================================

with tab2:
    st.subheader("📈 Температурные показания датчиков")

    st.markdown(
        """
        Здесь показаны температурные ряды по выбранным датчикам.  
        Красные точки обозначают моменты, которые система определила как аномальные.
        """
    )

    # Дополнительные настройки графика внутри вкладки
    col_a, col_b = st.columns(2)

    with col_a:
        chart_sensors = st.multiselect(
            "Датчики на графике:",
            options=sorted(filtered_df["sensor_id"].unique()),
            default=sorted(filtered_df["sensor_id"].unique())
        )

    with col_b:
        show_anomaly_points = st.checkbox(
            "Показывать точки аномалий",
            value=True
        )

    chart_df = filtered_df[filtered_df["sensor_id"].isin(chart_sensors)].copy()

    if len(chart_df) == 0:
        st.warning("Выберите хотя бы один датчик для отображения графика.")
    else:
        fig = px.line(
            chart_df,
            x="timestamp",
            y="temperature_filled",
            color="sensor_id",
            title="Температурные тренды по выбранным датчикам",
            labels={
                "timestamp": "Время",
                "temperature_filled": "Температура, °C",
                "sensor_id": "Датчик"
            }
        )

        # Улучшение внешнего вида графика
        fig.update_layout(
            hovermode="x unified",
            legend_title_text="Датчик",
            xaxis_title="Время",
            yaxis_title="Температура, °C"
        )

        # Отображение аномалий поверх линий
        if show_anomaly_points:
            anomaly_points = chart_df[chart_df["final_anomaly"] == 1]

            fig.add_scatter(
                x=anomaly_points["timestamp"],
                y=anomaly_points["temperature_filled"],
                mode="markers",
                marker=dict(size=9, color="red"),
                name="Обнаруженные аномалии",
                text=anomaly_points["sensor_id"],
                hovertemplate=(
                    "Датчик: %{text}<br>"
                    "Время: %{x}<br>"
                    "Температура: %{y:.2f} °C<br>"
                    "<extra></extra>"
                )
            )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Краткая статистика по отображаемому графику
        chart_total_points = len(chart_df)
        chart_anomalies = int(chart_df["final_anomaly"].sum())
        chart_anomaly_percent = (
            chart_anomalies / chart_total_points * 100
            if chart_total_points > 0 else 0
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Точек на графике", chart_total_points)
        col2.metric("Аномалий на графике", chart_anomalies)
        col3.metric("Доля аномалий", f"{chart_anomaly_percent:.2f}%")


# ============================================================
# 8. ГРАФИК ANOMALY SCORE
# ============================================================
with tab3:
    st.subheader("🧠 Anomaly score")

    if active_mode == ANALYSIS_MODE_RULES_ONLY:
        st.info(
            "Anomaly score не рассчитывается: ИИ-модель в этом запуске "
            "не применялась."
        )
    else:
        st.markdown(
            """
            **Anomaly score** показывает степень подозрительности поведения датчика.  
            Чем ближе значение к **1**, тем более нетипичным считается участок.
        
            Условная интерпретация:
            - **0.00–0.60** — нормальное или слабо подозрительное поведение;
            - **0.60–0.85** — зона предупреждения;
            - **выше 0.85** — высокий риск.
            """
        )

        score_fig = px.line(
            filtered_df,
            x="timestamp",
            y="anomaly_score_norm",
            color="sensor_id",
            title="Оценка аномальности поведения датчиков",
            labels={
                "timestamp": "Время",
                "anomaly_score_norm": "Anomaly score",
                "sensor_id": "Датчик"
            }
        )

        # Линия уровня Warning
        score_fig.add_hline(
            y=0.60,
            line_dash="dash",
            annotation_text="Порог предупреждения",
            annotation_position="top left"
        )

        # Линия уровня High
        score_fig.add_hline(
            y=0.85,
            line_dash="dash",
            annotation_text="Высокий порог",
            annotation_position="top left"
        )

        # Точки финальных аномалий
        score_anomalies = filtered_df[filtered_df["final_anomaly"] == 1]

        score_fig.add_scatter(
            x=score_anomalies["timestamp"],
            y=score_anomalies["anomaly_score_norm"],
            mode="markers",
            marker=dict(size=8, color="red"),
            name="Финальные аномалии",
            text=score_anomalies["sensor_id"],
            hovertemplate=(
                "Датчик: %{text}<br>"
                "Время: %{x}<br>"
                "Anomaly score: %{y:.3f}<br>"
                "<extra></extra>"
            )
        )

        score_fig.update_layout(
            hovermode="x unified",
            yaxis_title="Anomaly score",
            xaxis_title="Время"
        )

        st.plotly_chart(score_fig, use_container_width=True)

# ============================================================
# 9. ЖУРНАЛ ТРЕВОГ
# ============================================================
with tab4:
    st.subheader("🚨 Журнал тревог")

    table_df = filtered_alarm_log.copy()

    if len(table_df) == 0:
        st.info("По выбранным фильтрам тревог не найдено.")
    else:
        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = table_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="⬇️ Скачать журнал тревог",
            data=csv_data,
            file_name="alarm_log_filtered.csv",
            mime="text/csv"
        )
    results_csv = filtered_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="⬇️ Скачать полный результат анализа",
        data=results_csv,
        file_name="temperature_anomaly_results_filtered.csv",
        mime="text/csv"
    )
    # ============================================================
    # 10. СВОДКА ПО ТИПАМ СОБЫТИЙ
    # ============================================================

    st.subheader("📊 Сводка по типам событий")

    event_counts = (
        filtered_alarm_log["Тип_события"]
        .value_counts()
        .reset_index()
    )

    event_counts.columns = ["Тип события", "Количество"]

    if len(event_counts) > 0:
        bar_fig = px.bar(
            event_counts,
            x="Тип события",
            y="Количество",
            title="Количество тревог по типам событий"
        )

        st.plotly_chart(bar_fig, use_container_width=True)
    else:
        st.info("Нет данных для построения сводки.")


# ============================================================
# 11. ПОЯСНЕНИЕ ДЛЯ ЗАЩИТЫ
# ============================================================
with tab5:
    st.subheader("ℹ️ Описание работы MVP")

    st.markdown(
    """
    ### Режим анализа пользовательских данных

    Пользователь может загрузить собственный CSV-файл с температурными данными.
    Файл должен содержать колонки:

    - `timestamp` — дата и время измерения;
    - `sensor_id` — идентификатор датчика;
    - `temperature` — температура.

    После загрузки система выполняет предобработку, рассчитывает признаки,
    применяет правила и модель Isolation Forest, а затем формирует журнал тревог.

    Если пользовательские данные не содержат колонку `scenario`,
    система автоматически помечает их как `user_data`.
    """
)
