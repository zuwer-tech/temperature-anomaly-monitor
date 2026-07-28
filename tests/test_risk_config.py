import pytest

from risk_config import (
    RISK_MATRIX,
    RISK_RANK,
    RULE_RISK_LEVEL,
    assess_risk,
)


def test_risk_matrix_has_ordered_criteria_and_recommendations():
    assert list(RISK_MATRIX) == ["Normal", "Warning", "Medium", "High"]
    assert RISK_RANK == {"Normal": 0, "Warning": 1, "Medium": 2, "High": 3}
    for definition in RISK_MATRIX.values():
        assert definition["criteria"].strip()
        assert definition["recommendation"].strip()


def test_every_engineering_rule_has_expected_level():
    assert RULE_RISK_LEVEL == {
        "signal_loss": "Medium",
        "sharp_jump": "High",
        "z_score": "Warning",
        "stuck_sensor": "Medium",
        "group_deviation": "Warning",
        "sustained_overheat": "Warning",
    }


def test_highest_risk_wins_for_simultaneous_rules():
    level, recommendation = assess_risk(
        ["signal_loss", "sharp_jump", "sustained_overheat"]
    )
    assert level == "High"
    assert recommendation == RISK_MATRIX["High"]["recommendation"]


def test_ml_score_can_only_raise_an_existing_alarm():
    warning, _ = assess_risk([], iforest_anomaly=True, anomaly_score_norm=0.4,
                             medium_threshold=0.6, high_threshold=0.85)
    medium, _ = assess_risk([], iforest_anomaly=True, anomaly_score_norm=0.7,
                            medium_threshold=0.6, high_threshold=0.85)
    high, _ = assess_risk(["z_score"], anomaly_score_norm=0.9,
                          medium_threshold=0.6, high_threshold=0.85)
    assert (warning, medium, high) == ("Warning", "Medium", "High")


def test_ml_score_does_not_create_alarm_without_detection_evidence():
    level, _ = assess_risk([], anomaly_score_norm=1.0,
                           medium_threshold=0.6, high_threshold=0.85)
    assert level == "Normal"


def test_unknown_rule_is_rejected_instead_of_silently_downgraded():
    with pytest.raises(ValueError, match="unknown_rule"):
        assess_risk(["unknown_rule"])