"""Проверки воспроизводимой политики зависимостей."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def _requirement_lines(filename):
    return [
        line.split("#", 1)[0].strip()
        for line in (ROOT / filename).read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    ]


def _package_name(requirement):
    name = re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0]
    return name.lower().replace("_", "-")


def test_direct_dependencies_have_lower_and_upper_bounds():
    requirements = [
        *_requirement_lines("requirements.txt"),
        *_requirement_lines("requirements-dev.txt"),
    ]

    assert requirements
    for requirement in requirements:
        assert ">=" in requirement, requirement
        assert "<" in requirement, requirement
        assert "==" not in requirement, requirement


def test_constraints_exactly_pin_and_cover_direct_dependencies():
    constraints = _requirement_lines("constraints.txt")
    exact_versions = {}

    for constraint in constraints:
        assert re.fullmatch(r"[A-Za-z0-9_.-]+==[^=,]+", constraint), constraint
        name, version = constraint.split("==", 1)
        exact_versions[name.lower().replace("_", "-")] = version

    direct_names = {
        _package_name(requirement)
        for filename in ("requirements.txt", "requirements-dev.txt")
        for requirement in _requirement_lines(filename)
    }
    assert direct_names <= exact_versions.keys()
    assert "pip" in exact_versions
    assert len(exact_versions) > len(direct_names) * 3


def test_unused_seaborn_is_not_installed():
    all_requirements = "\n".join(
        requirement
        for filename in (
            "requirements.txt",
            "requirements-dev.txt",
            "constraints.txt",
        )
        for requirement in _requirement_lines(filename)
    ).lower()
    assert "seaborn" not in all_requirements


def test_ci_uses_constraints_pip_check_and_key_pipeline():
    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(
        encoding="utf-8"
    )

    assert 'python-version: "3.12"' in workflow
    assert "-c constraints.txt" in workflow
    assert "python -m pip check" in workflow
    for command in (
        "preprocessing.py",
        "train_model.py",
        "anomaly_detection.py",
        "evaluation.py",
    ):
        assert command in workflow


def test_readme_documents_both_installation_modes():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "Нужен Python 3.12" in readme
    assert "python -m pip install -r requirements.txt" in readme
    assert (
        "python -m pip install -r requirements.txt -c constraints.txt"
        in readme
    )

def test_coverage_config_and_ci_report_branches():
    dev_requirements = _requirement_lines("requirements-dev.txt")
    assert any(
        _package_name(requirement) == "pytest-cov"
        for requirement in dev_requirements
    )

    coverage_config = (ROOT / ".coveragerc").read_text(encoding="utf-8")
    assert "branch = True" in coverage_config
    assert "fail_under = 50" in coverage_config
    for omitted_path in ("tests/*", "notebooks/*", "pages/*"):
        assert omitted_path in coverage_config

    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(
        encoding="utf-8"
    )
    for option in (
        "--cov=.",
        "--cov-branch",
        "--cov-report=term-missing",
        "--cov-report=json:coverage.json",
        "--cov-fail-under=50",
    ):
        assert option in workflow
    assert "actions/upload-artifact@v4" in workflow

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "coverage.json" in gitignore
