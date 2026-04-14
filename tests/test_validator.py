import pytest
from src.data.validator import validate_patient


def test_valid_data_passes():
    """Clean record within all normal ranges must pass with no flags."""
    data = {'hemoglobin': 14.2, 'glucose': 90, 'wbc': 8000, 'platelets': 200000, 'creatinine': 1.0}
    result = validate_patient(data)
    assert result is not None
    assert result.warning_flags == []


def test_absolute_violation_rejected():
    """Biologically impossible value must be hard rejected."""
    data = {'hemoglobin': 150.0, 'glucose': 90, 'wbc': 8000, 'platelets': 200000, 'creatinine': 1.0}
    result = validate_patient(data)
    assert result is None


def test_missing_field_rejected():
    """Record missing a required field must be rejected."""
    data = {'hemoglobin': 14.2, 'wbc': 8000, 'platelets': 200000, 'creatinine': 1.0}  # glucose missing
    result = validate_patient(data)
    assert result is None


def test_critical_range_accepted_with_warning():
    """Value in critical range must be accepted but flagged - not rejected."""
    # Hb=25.0 is above critical_high (20.0) but below absolute_max (30.0)
    data = {'hemoglobin': 25.0, 'glucose': 90, 'wbc': 8000, 'platelets': 200000, 'creatinine': 1.0}
    result = validate_patient(data)
    assert result is not None
    assert 'hemoglobin_critical_range' in result.warning_flags
    assert result.warning_flags != []