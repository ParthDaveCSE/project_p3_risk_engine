from pydantic import BaseModel, model_validator, ValidationError
from typing import List
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()['parameters']


class PatientData(BaseModel):
    hemoglobin: float
    glucose: float
    wbc: float
    platelets: float
    creatinine: float
    warning_flags: List[str] = []  # populated during validation

    @model_validator(mode='after')
    def check_absolute_bounds(self):
        for field, value in self.model_dump().items():
            if field == 'warning_flags':
                continue

            rules = config[field]
            if value < rules['absolute_min'] or value > rules['absolute_max']:
                raise ValueError(
                    f'{field}={value} is outside biological limits '
                    f'absolute_min={rules["absolute_min"]}, '
                    f'absolute_max={rules["absolute_max"]}'
                )
        return self

    @model_validator(mode='after')
    def check_critical_range(self):
        flags = []
        for field, value in self.model_dump().items():
            if field == 'warning_flags':
                continue

            rules = config[field]
            if value < rules['critical_low'] or value > rules['critical_high']:
                flags.append(f'{field}_critical_range')
                logger.warning(
                    f'Critical range flag: {field}={value} '
                    f'critical_low={rules["critical_low"]}, '
                    f'critical_high={rules["critical_high"]}'
                )

        self.warning_flags = flags
        return self


def validate_patient(data: dict) -> PatientData | None:
    try:
        patient = PatientData(**data)

        if patient.warning_flags:
            logger.warning(f'Record accepted with flags: {patient.warning_flags}')
        else:
            logger.info('Record validated cleanly')

        return patient

    except ValidationError as e:
        logger.error(f'Hard rejection: {e.errors()[0]["msg"]}')
        return None


if __name__ == '__main__':
    print('\n--- Test 1: Valid record ---')
    valid = {'hemoglobin': 14.2, 'glucose': 90, 'wbc': 8000, 'platelets': 200000, 'creatinine': 1.0}
    result = validate_patient(valid)
    print(f'Result: {result}\nWarning flags: {result.warning_flags if result else None}')

    print('\n--- Test 2: Absolute violation (Hb=150.0) ---')
    impossible = {'hemoglobin': 150.0, 'glucose': 90, 'wbc': 8000, 'platelets': 200000, 'creatinine': 1.0}
    result = validate_patient(impossible)
    print(f'Result: {result}')

    print('\n--- Test 3: Critical range (Hb=25.0) ---')
    dangerous = {'hemoglobin': 25.0, 'glucose': 90, 'wbc': 8000, 'platelets': 200000, 'creatinine': 1.0}
    result = validate_patient(dangerous)
    print(f'Result: {result}\nWarning flags: {result.warning_flags if result else None}')