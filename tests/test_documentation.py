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
