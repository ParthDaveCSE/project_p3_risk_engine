from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from src.data.validator import validate_patient
from src.data.confidence_scorer import compute_confidence
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PatientResult:
    patient_id: str
    status: str  # accepted, flagged, rejected, error
    confidence_score: Optional[float]
    confidence_interpretation: Optional[str]
    flagged_parameters: List[str]
    reason: Optional[str]
    data: Optional[Dict[str, Any]]

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "status": self.status,
            "confidence_score": self.confidence_score,
            "confidence_interpretation": self.confidence_interpretation,
            "flagged_parameters": self.flagged_parameters,
            "reason": self.reason,
            "data": self.data,
        }


@dataclass
class BatchResult:
    total: int
    accepted: List[PatientResult] = field(default_factory=list)
    flagged: List[PatientResult] = field(default_factory=list)
    rejected: List[PatientResult] = field(default_factory=list)
    errors: List[PatientResult] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Batch complete | Total: {self.total} | "
            f"Accepted: {len(self.accepted)} | Flagged: {len(self.flagged)} | "
            f"Rejected: {len(self.rejected)} | Errors: {len(self.errors)}"
        )


class PatientProcessor:
    def __init__(self):
        config = load_config()
        self.threshold = config.get("system_settings", {}).get("minimum_confidence_threshold", 0.65)
        logger.info(f"Pipeline initialized | Confidence threshold: {self.threshold}")

    def process_one(self, data: dict) -> PatientResult:
        """
        Process a single patient record through the full pipeline.
        Returns a PatientResult with status: accepted, flagged, or rejected.
        Never raises an exception - all failures are caught and returned as status=error records.
        """
        patient_id = str(data.get("patient_id", "UNKNOWN"))

        # Step 1: Validation (L5)
        patient = validate_patient(data)
        if patient is None:
            logger.warning(f"{patient_id} REJECTED - failed absolute bounds validation")
            return PatientResult(
                patient_id=patient_id,
                status="rejected",
                confidence_score=None,
                confidence_interpretation=None,
                flagged_parameters=[],
                reason="absolute_bounds_violation",
                data=None,
            )

        # Step 2: Confidence scoring (L6)
        try:
            report = compute_confidence(patient)
        except Exception as e:
            logger.error(f"{patient_id} Confidence scoring failed: {e}")
            return PatientResult(
                patient_id=patient_id,
                status="error",
                confidence_score=None,
                confidence_interpretation=None,
                flagged_parameters=[],
                reason=f"confidence_scoring_error: {str(e)}",
                data=None,
            )

        # Step 3: Triage decision
        if report.confidence_score < self.threshold:
            logger.warning(
                f"{patient_id} FLAGGED - confidence {report.confidence_score:.2f} "
                f"below threshold {self.threshold} | Flags: {report.flagged_parameters}"
            )
            return PatientResult(
                patient_id=patient_id,
                status="flagged",
                confidence_score=report.confidence_score,
                confidence_interpretation=report.interpretation,
                flagged_parameters=report.flagged_parameters,
                reason="low_confidence",
                data=None,
            )

        # Step 4: Accept
        logger.info(
            f"{patient_id} ACCEPTED - confidence {report.confidence_score:.2f} ({report.interpretation})"
        )
        return PatientResult(
            patient_id=patient_id,
            status="accepted",
            confidence_score=report.confidence_score,
            confidence_interpretation=report.interpretation,
            flagged_parameters=report.flagged_parameters,
            reason=None,
            data=patient.model_dump(exclude={"warning_flags"}),
        )

    def process_batch(self, records: List[dict]) -> BatchResult:
        """
        Process a batch of patient records with graceful degradation.
        One corrupt record never stops the rest.
        """
        result = BatchResult(total=len(records))

        for record in records:
            patient_id = str(record.get("patient_id", "UNKNOWN"))
            try:
                outcome = self.process_one(record)
                if outcome.status == "accepted":
                    result.accepted.append(outcome)
                elif outcome.status == "flagged":
                    result.flagged.append(outcome)
                elif outcome.status == "rejected":
                    result.rejected.append(outcome)
                else:
                    result.errors.append(outcome)
            except Exception as e:
                # Scenario 2: Unexpected exception for one record
                logger.error(
                    f"[{patient_id}] PIPELINE ERROR - unexpected exception: "
                    f"{type(e).__name__}: {e}"
                )
                result.errors.append(PatientResult(
                    patient_id=patient_id,
                    status="error",
                    confidence_score=None,
                    confidence_interpretation=None,
                    flagged_parameters=[],
                    reason=f"{type(e).__name__}: {str(e)}",
                    data=None,
                ))

        logger.info(result.summary())
        return result


if __name__ == "__main__":
    from src.utils.config_loader import validate_config_on_startup

    print("\n--- Startup Config Validation ---")
    validate_config_on_startup()
    print("Config validation passed -- pipeline starting")

    processor = PatientProcessor()

    print("\n--- Individual Record Processing ---")
    test_cases = [
        {"patient_id": "P-001", "hemoglobin": 14.2, "glucose": 90, "wbc": 8000, "platelets": 200000, "creatinine": 1.0},
        {"patient_id": "P-002", "hemoglobin": 9.0, "glucose": 90, "wbc": 8000, "platelets": 200000, "creatinine": 1.0},
        {"patient_id": "P-003", "hemoglobin": 29.0, "glucose": 90, "wbc": 8000, "platelets": 200000, "creatinine": 1.0},
        {"patient_id": "P-004", "hemoglobin": 150.0, "glucose": 90, "wbc": 8000, "platelets": 200000, "creatinine": 1.0},
    ]

    for case in test_cases:
        result = processor.process_one(case)
        print(f"{result.patient_id}: {result.status.upper()} | confidence: {result.confidence_score} | reason: {result.reason}")

    print("\n--- Batch Processing with Graceful Degradation ---")
    batch = test_cases + [
        {"patient_id": "P-005", "hemoglobin": "not_a_number", "glucose": 90, "wbc": 8000, "platelets": 200000, "creatinine": 1.0}
    ]
    batch_result = processor.process_batch(batch)
    print(batch_result.summary())
    print(f"Error records: {[e.patient_id for e in batch_result.errors]}")