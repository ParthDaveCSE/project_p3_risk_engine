from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd
import os
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


def export_accepted_records(batch_result: BatchResult, original_df=None,
                            output_path: str = "data/processed/clean_patients.csv") -> pd.DataFrame | None:
    """
    Export accepted records from a BatchResult to CSV.
    Only accepted records are exported – flagged and rejected records never reach the ML pipeline.
    If original_df is provided, labels are merged from the original data.
    """
    if not batch_result.accepted:
        logger.warning("No accepted records to export")
        return None

    records = []

    # Create a lookup dictionary for labels from original_df
    label_lookup = {}
    if original_df is not None and "label" in original_df.columns:
        # If patient_id exists in original_df, use it as lookup key
        if "patient_id" in original_df.columns:
            for idx, row in original_df.iterrows():
                pid = row.get("patient_id")
                if pid:
                    label_lookup[str(pid)] = row.get("label", 0)
        else:
            # Otherwise use index as fallback
            for idx, row in original_df.iterrows():
                label_lookup[str(idx)] = row.get("label", 0)

    for result in batch_result.accepted:
        if result.data:
            record = result.data.copy()
            record["patient_id"] = result.patient_id
            record["confidence_score"] = result.confidence_score

            # Add label from lookup dictionary if available
            if result.patient_id in label_lookup:
                record["label"] = label_lookup[result.patient_id]
            else:
                record["label"] = 0  # Default label

            records.append(record)

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Exported {len(df)} accepted records to {output_path}")
    return df


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
    from src.data.generator import generate_dataset

    print("\n--- Startup Config Validation ---")
    validate_config_on_startup()
    print("Config validation passed -- pipeline starting")

    # Generate synthetic data if clean_patients.csv doesn't exist
    if not os.path.exists("data/processed/clean_patients.csv"):
        print("\n--- Generating synthetic data for pipeline ---")
        df = generate_dataset(5000)

        # Store original dataframe with labels for later merging
        original_df = df.copy()

        # Add patient_id to original_df for lookup
        original_df["patient_id"] = [f"P-{idx:04d}" for idx in range(len(original_df))]

        # Convert to list of dicts with patient_id
        records = []
        for idx, row in df.iterrows():
            record = row.to_dict()
            record["patient_id"] = f"P-{idx:04d}"
            records.append(record)

        processor = PatientProcessor()
        batch_result = processor.process_batch(records)

        print("\n--- Exporting accepted records ---")
        export_accepted_records(batch_result, original_df=original_df)
    else:
        print("\n--- Clean patients file already exists ---")