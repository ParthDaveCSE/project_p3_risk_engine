"""
Clinical Risk Engine — Command Line Interface

Four commands:
    analyze   — Predict clinical risk (single patient or batch JSONL)
    validate  — Verify the clinical config YAML is valid
    status    — List all available model artifacts
    schema    — Show the required input field schema

Usage:
    uv run python src/cli/risk_engine.py analyze \\
        --data '{"hemoglobin": 9.2, "glucose": 180, "wbc": 12000,
                 "platelets": 95000, "creatinine": 4.1}'

    uv run python src/cli/risk_engine.py analyze \\
        --file patients.jsonl --output csv

    uv run python src/cli/risk_engine.py schema
    uv run python src/cli/risk_engine.py validate
    uv run python src/cli/risk_engine.py status
"""

import csv
import hashlib
import io
import json
import os
import time
from typing import Optional

import joblib
import pandas as pd
import typer

from src.utils.config_loader import load_config, validate_config_on_startup
from src.utils.logger import get_logger
from src.data.validator import validate_patient
from src.data.confidence_scorer import compute_confidence
from src.models.threshold_tuner import predict_with_threshold
from src.models.model_store import ModelArtifactNotFoundError

logger = get_logger(__name__)
config = load_config()
app = typer.Typer(
    name="risk-engine",
    help="Clinical Risk Engine — Predict, validate, and inspect.",
    add_completion=False,
)

CLINICAL_FIELDS = ["hemoglobin", "glucose", "wbc", "platelets", "creatinine"]

FIELD_SCHEMA = {
    "hemoglobin": {
        "type": "float", "unit": "g/dL",
        "description": "Hemoglobin concentration", "example": 14.2,
    },
    "glucose": {
        "type": "float", "unit": "mg/dL",
        "description": "Blood glucose level", "example": 90.0,
    },
    "wbc": {
        "type": "float", "unit": "/µL",
        "description": "White blood cell count", "example": 8000.0,
    },
    "platelets": {
        "type": "float", "unit": "/µL",
        "description": "Platelet count", "example": 200000.0,
    },
    "creatinine": {
        "type": "float", "unit": "mg/dL",
        "description": "Serum creatinine level", "example": 1.0,
    },
}

# Lazy singletons — loaded once per process, not on every call
_pipeline_singleton = None
_bundle_singleton = None
_explainer_singleton = None


def _current_config_hash(config_path: str = "config/clinical_rules.yaml") -> str:
    """Compute MD5 hash of current config file for drift detection."""
    if not os.path.exists(config_path):
        return "config_not_found"
    with open(config_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_pipeline():
    """Return feature pipeline, loading it once per process (lazy singleton)."""
    global _pipeline_singleton
    if _pipeline_singleton is None:
        model_dir = config["paths"]["model_dir"]
        pipeline_path = os.path.join(model_dir, "feature_pipeline.joblib")

        if not os.path.exists(pipeline_path):
            raise ModelArtifactNotFoundError(
                f"Feature pipeline not found: {pipeline_path} | "
                f"Run: uv run python src/features/feature_engineer.py"
            )

        _pipeline_singleton = joblib.load(pipeline_path)
        logger.info(f"Feature pipeline loaded (singleton): {pipeline_path}")

    return _pipeline_singleton


def get_bundle(bundle_name: str = "rf_balanced_production", version: str = "1.0.0"):
    """Return (model, threshold), loading the production bundle once per process."""
    global _bundle_singleton
    if _bundle_singleton is None:
        model_dir = config["paths"]["model_dir"]
        bundle_path = os.path.join(model_dir, f"{bundle_name}_bundle_v{version}.joblib")

        if os.path.exists(bundle_path):
            bundle = joblib.load(bundle_path)
            model = bundle["model"]
            threshold = bundle["threshold"]

            # Config drift detection
            current_hash = _current_config_hash()
            stored_hash = bundle.get("config_hash", None)

            if stored_hash and current_hash != stored_hash:
                logger.warning(
                    f"CONFIG DRIFT DETECTED | "
                    f"Bundle hash: {stored_hash[:8]}... | "
                    f"Current hash: {current_hash[:8]}... | "
                    f"Clinical rules changed since training."
                )
            elif stored_hash:
                logger.info(f"Config hash verified: no drift ({current_hash[:8]}...)")

            logger.info(f"Production bundle loaded: {bundle_path} | threshold={threshold}")
            _bundle_singleton = (model, threshold)

        else:
            # Fallback: load model and threshold separately
            logger.warning(f"Bundle not found at {bundle_path}. Falling back to separate artifacts.")
            from src.models.model_store import load_model
            from src.models.threshold_tuner import load_threshold_artifact

            model = load_model("random_forest_balanced", version)
            threshold_config = load_threshold_artifact(version)
            threshold = threshold_config["threshold"]
            _bundle_singleton = (model, threshold)

    return _bundle_singleton


def get_explainer(model):
    """Return SHAP TreeExplainer, building it once per process."""
    global _explainer_singleton
    if _explainer_singleton is None:
        import shap
        _explainer_singleton = shap.TreeExplainer(model)
        logger.info("SHAP TreeExplainer built (singleton)")
    return _explainer_singleton


def _apply_pipeline_to_patient(raw_input: dict, pipeline) -> pd.DataFrame:
    """Apply feature pipeline to a single patient's raw clinical values."""
    row = {field: [raw_input[field]] for field in CLINICAL_FIELDS}
    X = pd.DataFrame(row)
    X_transformed = pipeline.transform(X)

    if not isinstance(X_transformed, pd.DataFrame):
        cols = (
            X_transformed.columns.tolist()
            if hasattr(X_transformed, "columns")
            else [f"f{i}" for i in range(X_transformed.shape[1])]
        )
        X_transformed = pd.DataFrame(X_transformed, columns=cols)

    return X_transformed


def _check_pipeline_model_compatibility(model, X_transformed: pd.DataFrame):
    """Assert pipeline output shape matches model expectation."""
    if not hasattr(model, "n_features_in_"):
        return

    expected = model.n_features_in_
    actual = X_transformed.shape[1]

    if expected != actual:
        raise RuntimeError(
            f"Pipeline-model feature mismatch: "
            f"model expects {expected} features, pipeline produced {actual}. "
            f"Pipeline and model are from different training runs."
        )


def _log_inference(
    input_hash: str,
    y_pred: int,
    y_prob: float,
    confidence_score: float,
    threshold: float,
    latency_ms: float,
):
    """Log each inference for audit trail."""
    logger.info(
        f"INFERENCE | input_hash={input_hash[:12]} | "
        f"prediction={'HIGH-RISK' if y_pred == 1 else 'NORMAL'} | "
        f"probability={y_prob:.4f} | confidence={confidence_score:.4f} | "
        f"threshold={threshold:.4f} | latency_ms={latency_ms:.1f}"
    )


def _predict_single(raw_input: dict, explain: bool = False) -> dict:
    """Run the full inference pipeline for one patient."""
    start_ms = time.time() * 1000

    # Step 1: Validate
    patient = validate_patient(raw_input)
    if patient is None:
        return {
            "status": "rejected",
            "reason": "absolute_bounds_violation",
            "prediction": None,
            "probability": None,
            "confidence_score": None,
            "flagged_parameters": [],
        }

    # Step 2: Confidence
    confidence_report = compute_confidence(patient)

    # Step 3: Pipeline (singleton)
    pipeline = get_pipeline()
    patient_dict = patient.model_dump(exclude={"warning_flags"})
    X_transformed = _apply_pipeline_to_patient(patient_dict, pipeline)

    # Step 4: Compatibility assertion
    model, threshold = get_bundle()
    _check_pipeline_model_compatibility(model, X_transformed)

    # Step 5: Predict
    y_pred = predict_with_threshold(model, X_transformed, threshold)
    y_prob = float(model.predict_proba(X_transformed)[:, 1][0])

    # Step 6: SHAP (optional)
    shap_explanation = None
    if explain:
        try:
            explainer = get_explainer(model)
            shap_values = explainer.shap_values(X_transformed)
            from src.explainability.shap_explainer import get_local_explanation
            shap_explanation = get_local_explanation(
                shap_values, explainer, X_transformed, patient_index=0
            )
        except Exception as e:
            logger.warning(f"SHAP explanation unavailable: {e}")

    # Step 7: Audit log
    input_hash = hashlib.md5(
        json.dumps(raw_input, sort_keys=True).encode()
    ).hexdigest()
    latency_ms = time.time() * 1000 - start_ms
    _log_inference(
        input_hash, int(y_pred[0]), y_prob,
        confidence_report.confidence_score, threshold, latency_ms
    )

    return {
        "status": "accepted",
        "prediction": "HIGH-RISK" if int(y_pred[0]) == 1 else "NORMAL",
        "probability": round(y_prob, 4),
        "threshold": threshold,
        "confidence_score": confidence_report.confidence_score,
        "confidence_level": confidence_report.interpretation,
        "flagged_parameters": confidence_report.flagged_parameters,
        "latency_ms": round(latency_ms, 1),
        "shap_explanation": shap_explanation,
        "input": raw_input,
    }


def _format_text_output(result: dict) -> str:
    """Format result as human-readable text."""
    if result["status"] == "rejected":
        return (
            "\n  REJECTED — Input data failed clinical validation.\n"
            "  One or more values are outside absolute biological limits.\n"
        )

    y_prob = result["probability"]
    conf_bar = "█" * int(y_prob * 10) + "░" * (10 - int(y_prob * 10))

    lines = [
        "", "=" * 58,
        "  CLINICAL RISK ENGINE — PATIENT ASSESSMENT",
        "=" * 58,
        "", "  INPUT CLINICAL VALUES", "  " + "-" * 54,
    ]
    for field in CLINICAL_FIELDS:
        lines.append(f"  {field:<20} {result['input'].get(field, 'N/A')}")

    lines += [
        "", "  DATA QUALITY", "  " + "-" * 54,
        f"  Confidence score:  {result['confidence_score']:.3f}  ({result['confidence_level']})",
    ]
    if result["flagged_parameters"]:
        lines.append(f"  Critical flags:    {', '.join(result['flagged_parameters'])}")
    else:
        lines.append("  Critical flags:    None")

    lines += [
        "", "  RISK PREDICTION", "  " + "-" * 54,
        f"  Result:            {result['prediction']}",
        f"  Probability:       {y_prob:.4f}  [{conf_bar}]",
        f"  Threshold:         {result['threshold']:.4f}",
        f"  Latency:           {result['latency_ms']:.1f} ms",
        "", "=" * 58, "",
    ]
    return "\n".join(lines)


def _format_json_output(result: dict) -> str:
    """Format result as JSON."""
    out = {
        "status": result["status"],
        "prediction": result.get("prediction"),
        "probability": result.get("probability"),
        "threshold": result.get("threshold"),
        "confidence_score": result.get("confidence_score"),
        "confidence_level": result.get("confidence_level"),
        "flagged_parameters": result.get("flagged_parameters", []),
        "latency_ms": result.get("latency_ms"),
    }
    if result.get("shap_explanation"):
        out["shap_contributions"] = result["shap_explanation"]["contributions"]
    return json.dumps(out)


def _result_to_csv_row(result: dict) -> dict:
    """Convert result to CSV row dictionary."""
    row = {
        "status": result.get("status"),
        "prediction": result.get("prediction"),
        "probability": result.get("probability"),
        "threshold": result.get("threshold"),
        "confidence_score": result.get("confidence_score"),
        "confidence_level": result.get("confidence_level"),
        "flagged_parameters": ";".join(result.get("flagged_parameters", [])),
        "latency_ms": result.get("latency_ms"),
    }
    for field in CLINICAL_FIELDS:
        row[field] = result.get("input", {}).get(field, "")
    return row


# ── CLI Commands ───────────────────────────────────────────────────────────

@app.command()
def analyze(
    data: Optional[str] = typer.Option(
        None, "--data", "-d",
        help="Patient clinical values as a JSON string.",
    ),
    file: Optional[str] = typer.Option(
        None, "--file", "-f",
        help="JSON file (single patient) or JSONL file (batch).",
    ),
    bundle_name: str = typer.Option(
        "rf_balanced_production", "--bundle", "-b",
        help="Production bundle name.",
    ),
    version: str = typer.Option(
        "1.0.0", "--version", "-v",
        help="Production bundle version.",
    ),
    explain: bool = typer.Option(
        False, "--explain", "-e",
        help="Include SHAP explanation.",
    ),
    output: str = typer.Option(
        "text", "--output", "-o",
        help="Output format: 'text', 'json', or 'csv'.",
    ),
):
    """Predict clinical risk for a single patient or a batch."""
    if output not in ("text", "json", "csv"):
        typer.echo("Error: --output must be 'text', 'json', or 'csv'", err=True)
        raise typer.Exit(code=1)

    if data is None and file is None:
        typer.echo("Error: provide patient data with --data or --file.", err=True)
        raise typer.Exit(code=1)

    # Load singletons eagerly so artifact errors are reported early
    try:
        get_pipeline()
        get_bundle(bundle_name, version)
    except ModelArtifactNotFoundError as e:
        typer.echo(f"\n  Error: {e}", err=True)
        raise typer.Exit(code=3)

    # Parse inputs
    records = []
    if file is not None:
        if not os.path.exists(file):
            typer.echo(f"Error: file not found: {file}", err=True)
            raise typer.Exit(code=1)
        with open(file, "r") as fh:
            content = fh.read().strip()
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        if len(lines) > 1:
            for i, line in enumerate(lines):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    typer.echo(f"Error: invalid JSON on line {i+1}: {e}", err=True)
                    raise typer.Exit(code=1)
        else:
            try:
                records.append(json.loads(content))
            except json.JSONDecodeError as e:
                typer.echo(f"Error: invalid JSON in file: {e}", err=True)
                raise typer.Exit(code=1)
    else:
        try:
            records.append(json.loads(data))
        except json.JSONDecodeError as e:
            typer.echo(f"Error: invalid JSON in --data: {e}", err=True)
            raise typer.Exit(code=1)

    # Process records
    results = []
    has_rejection = False

    for record in records:
        missing = [f for f in CLINICAL_FIELDS if f not in record]
        if missing:
            typer.echo(f"Error: missing required fields: {missing}", err=True)
            raise typer.Exit(code=1)

        try:
            result = _predict_single(record, explain=explain)
        except RuntimeError as e:
            typer.echo(f"\n  System error: {e}", err=True)
            raise typer.Exit(code=3)

        results.append(result)
        if result["status"] == "rejected":
            has_rejection = True

    # Output
    if output == "text":
        for result in results:
            typer.echo(_format_text_output(result))
    elif output == "json":
        if len(results) == 1:
            typer.echo(_format_json_output(results[0]))
        else:
            typer.echo(json.dumps([json.loads(_format_json_output(r)) for r in results]))
    elif output == "csv":
        rows = [_result_to_csv_row(r) for r in results]
        if rows:
            writer_buf = io.StringIO()
            writer = csv.DictWriter(writer_buf, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
            typer.echo(writer_buf.getvalue().rstrip())

    if has_rejection and len(results) == 1:
        raise typer.Exit(code=2)


@app.command()
def validate(
    config_path: str = typer.Option(
        "config/clinical_rules.yaml", "--config", "-c",
        help="Path to the clinical rules YAML config file.",
    ),
):
    """Validate the clinical config YAML."""
    typer.echo(f"\n  Validating config: {config_path}")
    try:
        cfg = validate_config_on_startup(config_path)
        typer.echo(
            f"\n  ✓ Config valid\n"
            f"  Parameters:    {list(cfg['parameters'].keys())}\n"
            f"  Config hash:   {_current_config_hash(config_path)[:16]}...\n"
        )
    except SystemExit:
        typer.echo(f"\n  ✗ Config validation FAILED: {config_path}\n", err=True)
        raise typer.Exit(code=1)
    except FileNotFoundError:
        typer.echo(f"\n  ✗ Config file not found: {config_path}\n", err=True)
        raise typer.Exit(code=1)


@app.command()
def status(
    model_dir: Optional[str] = typer.Option(
        None, "--dir",
        help="Artifact directory to scan.",
    ),
    output: str = typer.Option("text", "--output", "-o", help="'text' or 'json'."),
):
    """List all available model artifacts."""
    import src.models.model_store as ms

    if model_dir:
        orig = ms.config.copy()
        ms.config = {**orig, "paths": {**orig["paths"], "model_dir": model_dir}}

    artifacts = ms.list_available_models()

    if not artifacts:
        typer.echo("\n  No model artifacts found.\n")
        return

    # Mark latest versions
    latest_per_name = {}
    for a in artifacts:
        ver = tuple(int(x) for x in a["version"].split("."))
        if a["name"] not in latest_per_name or ver > latest_per_name[a["name"]]:
            latest_per_name[a["name"]] = ver

    for a in artifacts:
        ver = tuple(int(x) for x in a["version"].split("."))
        a["is_latest"] = (ver == latest_per_name.get(a["name"], (0, 0, 0)))

    if output == "json":
        typer.echo(json.dumps(artifacts, indent=2))
        return

    typer.echo(f"\n  {'Name':<38} {'Type':<8} {'Ver':<8} {'KB':>6}  Latest")
    typer.echo("  " + "-" * 70)
    for a in artifacts:
        latest_mark = "✓" if a.get("is_latest") else ""
        typer.echo(
            f"  {a['name']:<38} {a.get('type', 'model'):<8} "
            f"{a['version']:<8} {a.get('size_kb', 0):>6.1f}  {latest_mark}"
        )
    typer.echo(f"\n  Total: {len(artifacts)} artifact(s)")


@app.command()
def schema(
    output: str = typer.Option("text", "--output", "-o", help="'text' or 'json'."),
):
    """Show the required input field schema."""
    if output == "json":
        typer.echo(json.dumps(FIELD_SCHEMA, indent=2))
        return

    typer.echo(f"\n  {'Field':<20} {'Type':<8} {'Unit':<12} {'Example':>10}  Description")
    typer.echo("  " + "-" * 70)
    for field, info in FIELD_SCHEMA.items():
        typer.echo(
            f"  {field:<20} {info['type']:<8} {info['unit']:<12} "
            f"{str(info['example']):>10}  {info['description']}"
        )
    typer.echo("\n  All fields required. All values must be floats.\n")


if __name__ == "__main__":
    app()