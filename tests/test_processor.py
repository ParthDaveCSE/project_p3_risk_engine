import pytest
from src.pipeline.processor import PatientProcessor
from src.utils.config_loader import validate_config_on_startup


def make_valid_record(patient_id="P-TEST"):
    return {
        "patient_id": patient_id,
        "hemoglobin": 14.2,
        "glucose": 90,
        "wbc": 8000,
        "platelets": 200000,
        "creatinine": 1.0,
    }


def test_valid_patient_accepted():
    """Clean record must be accepted with confidence 1.0."""
    processor = PatientProcessor()
    result = processor.process_one(make_valid_record())
    assert result.status == "accepted"
    assert result.confidence_score == 1.0
    assert result.reason is None


def test_absolute_violation_rejected():
    """Record with impossible value must be rejected."""
    processor = PatientProcessor()
    record = make_valid_record()
    record["hemoglobin"] = 150.0
    result = processor.process_one(record)
    assert result.status == "rejected"
    assert result.confidence_score is None
    assert result.reason == "absolute_bounds_violation"


def test_sick_patient_accepted_not_flagged():
    """Anemic patient with Hemoglobin 9.0 must be accepted, not flagged."""
    processor = PatientProcessor()
    record = make_valid_record()
    record["hemoglobin"] = 9.0
    result = processor.process_one(record)
    assert result.status == "accepted"
    assert result.confidence_score == 1.0


def test_extreme_value_accepted_or_flagged():
    """Patient with near-impossible value (Hb=29.0) has confidence ~0.82.
    If threshold is 0.80, it's accepted. If threshold is higher, it's flagged.
    This test checks that the status is one of the two valid states."""
    processor = PatientProcessor()
    record = make_valid_record()
    record["hemoglobin"] = 29.0
    result = processor.process_one(record)
    # Both accepted and flagged are valid depending on threshold setting
    assert result.status in ["accepted", "flagged"]
    assert result.confidence_score is not None
    assert result.confidence_score < 0.9  # Should be reduced from 1.0


def test_batch_continues_after_error():
    """One corrupt record in a batch must not stop the rest.
    The corrupt record should be either rejected or in errors."""
    processor = PatientProcessor()
    batch = [
        make_valid_record("P-001"),
        make_valid_record("P-002"),
        {
            "patient_id": "P-BAD",
            "hemoglobin": "not_a_number",
            "glucose": 90,
            "wbc": 8000,
            "platelets": 200000,
            "creatinine": 1.0,
        },
        make_valid_record("P-004"),
    ]
    result = processor.process_batch(batch)
    assert result.total == 4
    assert len(result.accepted) == 3
    # The bad record is either rejected or in errors
    assert len(result.rejected) + len(result.errors) == 1


def test_patient_id_in_all_results():
    """Every result must carry the patient_id for traceability."""
    processor = PatientProcessor()
    record = make_valid_record("P-TRACE-001")
    result = processor.process_one(record)
    assert result.patient_id == "P-TRACE-001"


def test_config_validation_passes():
    """Startup config validation must pass on the current config."""
    config = validate_config_on_startup()
    assert "parameters" in config
    assert "system_settings" in config