from src.data.validator import validate_patient
from src.data.confidence_scorer import compute_confidence, score_parameter


def test_normal_patient_scores_one():
    """Clean record within all normal ranges must score exactly 1.0."""
    patient = validate_patient({"hemoglobin": 14.2, "glucose": 90, "wbc": 8000, "platelets": 200000, "creatinine": 1.0})
    assert patient is not None
    report = compute_confidence(patient)
    assert report.confidence_score == 1.0
    assert report.interpretation == "high"
    assert report.flagged_parameters == []


def test_sick_patient_scores_one():
    """Sick patient with values in critical range must score 1.0."""
    patient = validate_patient({"hemoglobin": 9.0, "glucose": 55, "wbc": 8000, "platelets": 200000, "creatinine": 1.0})
    assert patient is not None
    report = compute_confidence(patient)
    assert report.confidence_score == 1.0
    assert report.interpretation == "high"


def test_extreme_value_scores_below_threshold():
    """Value near absolute limit must score significantly below 1.0."""
    patient = validate_patient({"hemoglobin": 29.0, "glucose": 90, "wbc": 8000, "platelets": 200000, "creatinine": 1.0})
    assert patient is not None
    report = compute_confidence(patient)
    assert report.parameter_scores["hemoglobin"] < 0.2
    assert report.confidence_score < 1.0


def test_parameter_score_zone_logic():
    """Unit test the zone scoring function directly."""
    from src.utils.config_loader import load_config
    rules = load_config()["parameters"]["hemoglobin"]
    mid_critical = (rules["critical_low"] + rules["critical_high"]) / 2
    assert score_parameter("hemoglobin", mid_critical) == 1.0
    assert score_parameter("hemoglobin", rules["critical_low"]) == 1.0
    mid_extreme = (rules["absolute_min"] + rules["critical_low"]) / 2
    assert 0.4 < score_parameter("hemoglobin", mid_extreme) < 0.6


def test_report_carries_warning_flags_from_validator():
    """Warning flags set by L5 validator must appear in the confidence report."""
    # Use Hb=25.0 which is above critical_high (20.0) to trigger warning flag
    patient = validate_patient({"hemoglobin": 25.0, "glucose": 90, "wbc": 8000, "platelets": 200000, "creatinine": 1.0})
    assert patient is not None
    assert "hemoglobin_critical_range" in patient.warning_flags
    report = compute_confidence(patient)
    assert "hemoglobin_critical_range" in report.flagged_parameters