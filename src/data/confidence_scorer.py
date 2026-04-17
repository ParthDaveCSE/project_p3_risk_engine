import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.data.validator import validate_patient

logger = get_logger(__name__)
config = load_config()
parameters_config = config["parameters"]
confidence_config = config["confidence"]

CLINICAL_FIELDS = {"hemoglobin", "glucose", "wbc", "platelets", "creatinine"}


@dataclass
class ConfidenceReport:
    confidence_score: float
    parameter_scores: Dict[str, float]
    flagged_parameters: List[str]
    interpretation: str
    recommendation: str

    def summary(self) -> str:
        return (
            f"Confidence: {self.confidence_score:.2f} ({self.interpretation}) | "
            f"Flagged: {self.flagged_parameters if self.flagged_parameters else 'none'}"
        )


def score_parameter(field_name: str, value: float) -> float:
    """
    Score a single parameter value based on its distance from biological limits.
    Returns 1.0 if the value is within the critical range (disease is trusted).
    Returns a decaying score as the value approaches absolute limits.
    """
    rules = parameters_config[field_name]
    crit_low = rules["critical_low"]
    crit_high = rules["critical_high"]
    abs_low = rules["absolute_min"]
    abs_high = rules["absolute_max"]

    # Zone 1: Within critical range - full confidence
    if crit_low <= value <= crit_high:
        return 1.0

    # Zone 2: Between absolute minimum and critical low
    elif abs_low <= value < crit_low:
        gap = crit_low - abs_low
        distance_from_floor = value - abs_low
        return round(distance_from_floor / (gap + 1e-9), 4)

    # Zone 3: Between critical high and absolute maximum
    elif crit_high < value <= abs_high:
        gap = abs_high - crit_high
        distance_from_ceiling = abs_high - value
        return round(distance_from_ceiling / (gap + 1e-9), 4)

    # Zone 4: Outside absolute limits - should never reach here
    else:
        logger.error(
            f"PIPELINE BREACH: {field_name}={value} reached confidence scorer "
            f"despite being outside absolute limits. Validator may have failed."
        )
        return 0.0


def compute_confidence(patient) -> ConfidenceReport:
    """Compute a structured confidence report for a validated patient record."""
    parameter_scores = {}
    for field_name, value in patient.model_dump().items():
        if field_name not in CLINICAL_FIELDS:
            continue
        parameter_scores[field_name] = score_parameter(field_name, value)

    overall_score = round(float(np.mean(list(parameter_scores.values()))), 4)
    flagged = patient.warning_flags

    high_threshold = confidence_config["high_threshold"]
    moderate_threshold = confidence_config["moderate_threshold"]
    interpretations = confidence_config["interpretations"]

    if overall_score >= high_threshold:
        interpretation, recommendation = "high", interpretations["high"]
    elif overall_score >= moderate_threshold:
        interpretation, recommendation = "moderate", interpretations["moderate"]
    else:
        interpretation, recommendation = "low", interpretations["low"]

    report = ConfidenceReport(
        confidence_score=overall_score,
        parameter_scores=parameter_scores,
        flagged_parameters=flagged,
        interpretation=interpretation,
        recommendation=recommendation
    )

    logger.info(report.summary())
    return report


if __name__ == "__main__":
    print("\n--- Patient A: Sick but reliable (anemia) ---")
    # FIXED: Changed "plates" to "platelets"
    patient_a = validate_patient({"hemoglobin": 9.0, "glucose": 90, "wbc": 8000, "platelets": 200000, "creatinine": 1.0})
    if patient_a:
        report_a = compute_confidence(patient_a)
        print(f"Score: {report_a.confidence_score}")
        print(f"Interpretation: {report_a.interpretation}")
        print(f"Parameter scores: {report_a.parameter_scores}")

    print("\n--- Patient B: Near biological ceiling (suspicious) ---")
    # FIXED: Changed "plates" to "platelets"
    patient_b = validate_patient({"hemoglobin": 29.0, "glucose": 90, "wbc": 8000, "platelets": 200000, "creatinine": 1.0})
    if patient_b:
        report_b = compute_confidence(patient_b)
        print(f"Score: {report_b.confidence_score}")
        print(f"Interpretation: {report_b.interpretation}")
        print(f"Parameter scores: {report_b.parameter_scores}")

    print("\n--- Patient C: Multiple critical flags ---")
    # FIXED: Changed "plates" to "platelets"
    patient_c = validate_patient({"hemoglobin": 9.0, "glucose": 45, "wbc": 8000, "platelets": 200000, "creatinine": 1.5})
    if patient_c:
        report_c = compute_confidence(patient_c)
        print(f"Score: {report_c.confidence_score}")
        print(f"Interpretation: {report_c.interpretation}")
        print(f"Parameter scores: {report_c.parameter_scores}")