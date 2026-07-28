from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "module_path",
    [
        "data_quality.py",
        "events.py",
        "evaluation.py",
        "model_schema.py",
        "rule_config.py",
    ],
)
def test_current_documented_modules_exist(module_path):
    assert (ROOT / module_path).is_file()


def test_architecture_does_not_describe_removed_online_training():
    architecture = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")

    assert "обучается на лету" not in architecture
    assert "fit на лету" not in architecture
    assert "Анализируемый CSV никогда не вызывает" in architecture
    assert "ModelNotTrainedError" in architecture


def test_readme_links_current_status_document():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "[docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)" in readme
    assert "python -m pytest -q" in readme
    assert "rules-only" in readme
    assert "rules+ML" in readme



def test_technical_report_has_required_sections():
    report = (ROOT / "docs" / "TECHNICAL_REPORT.md").read_text(encoding="utf-8")

    required_headings = [
        "## 1. Задача и назначение",
        "## 2. Данные",
        "## 3. Архитектура",
        "## 4. Методы обнаружения",
        "## 5. Обучение и независимая оценка",
        "## 6. Результаты",
        "## 7. Ограничения",
        "## 8. Воспроизводимость",
        "## 9. Направления развития",
    ]
    for heading in required_headings:
        assert heading in report

    assert "не является вероятностью аварии" in report
    assert "evaluation также не вызывает" in report


def test_defense_materials_support_repeatable_demo():
    presentation = ROOT / "presentation" / "temperature-anomaly-monitor-defense.pptx"
    outline = (ROOT / "docs" / "PRESENTATION.md").read_text(encoding="utf-8")
    demo = (ROOT / "docs" / "DEMO_SCRIPT.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert presentation.is_file()
    assert presentation.read_bytes()[:2] == b"PK"
    assert "Структура 8 слайдов" in outline
    for step in ["качество", "график", "alarm", "объяснение", "event log", "ограничения"]:
        assert step in demo
    assert "5–10 минут" in demo
    assert "Резервный сценарий" in demo
    assert "[docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md)" in readme
    assert "[docs/PRESENTATION.md](docs/PRESENTATION.md)" in readme
