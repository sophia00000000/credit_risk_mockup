"""API FastAPI para el mockup ejecutivo de riesgo crediticio."""
from pathlib import Path
import json
import math
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "gbm_bundle.joblib"
INFO_PATH = BASE_DIR / "model_info.json"
app = FastAPI(title="Cuidarte Credit Risk — GBM", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(BASE_DIR / "index.html")


class LoanApplication(BaseModel):
    person_age: int = Field(ge=18, le=100)
    person_income: float = Field(gt=0)
    person_home_ownership: str
    person_emp_length: float = Field(ge=0, le=50)
    loan_intent: str
    loan_grade: str
    loan_amnt: float = Field(gt=0)
    loan_int_rate: float = Field(ge=0, le=100)
    cb_person_default_on_file: str
    cb_person_cred_hist_length: float = Field(ge=0, le=100)


def load_assets():
    if not MODEL_PATH.exists() or not INFO_PATH.exists():
        raise HTTPException(status_code=503, detail="El modelo aún no está entrenado. Sigue los pasos de README.md para descargar el CSV y ejecutar train_model.py.")
    return joblib.load(MODEL_PATH), json.loads(INFO_PATH.read_text(encoding="utf-8"))

@app.get("/health")
def health():
    ready = MODEL_PATH.exists() and INFO_PATH.exists()
    return {"status": "ok", "model_ready": ready, "model": "Gradient Boosting"}

@app.get("/model-info")
def model_info():
    _, info = load_assets()
    return info

@app.post("/predict")
def predict(application: LoanApplication):
    bundle, info = load_assets()
    d = application.model_dump() if hasattr(application, "model_dump") else application.dict()
    # Derivación usada en el dataset: proporción entre monto solicitado e ingreso anual.
    d["loan_percent_income"] = d["loan_amnt"] / d["person_income"]
    X_one = pd.get_dummies(pd.DataFrame([d])).astype(float)
    X_one = X_one.reindex(columns=bundle["columns"], fill_value=0.0)
    probability = float(bundle["model"].predict_proba(X_one)[0, 1])
    if not math.isfinite(probability):
        raise HTTPException(status_code=500, detail="El modelo devolvió una probabilidad inválida.")
    low, high = float(bundle["t_low"]), float(bundle["t_high"])
    if probability < low:
        level, verdict, action = "BAJO", "APROBACIÓN AUTOMÁTICA PROPUESTA", "approval"
        explanation = "La probabilidad estimada está por debajo del umbral de aprobación. La recomendación es continuar con el flujo automático, sujeto a las políticas y verificaciones del banco."
    elif probability >= high:
        level, verdict, action = "ALTO", "RECHAZO AUTOMÁTICO PROPUESTO", "reject"
        explanation = "La probabilidad estimada alcanza el umbral de alto riesgo. La solicitud se marca para rechazo propuesto y debe respetar las políticas de explicación y reconsideración de la entidad."
    else:
        level, verdict, action = "MEDIO", "REVISIÓN MANUAL", "review"
        explanation = "El resultado se encuentra en la zona intermedia. El modelo no debe resolver el caso por sí solo: un analista debe revisar la capacidad de pago y la información complementaria."
    return {
        "probability_default": probability,
        "risk_level": level,
        "verdict": verdict,
        "action": action,
        "explanation": explanation,
        "loan_percent_income": d["loan_percent_income"],
        "thresholds": {"approval_below": low, "reject_at_or_above": high},
        "top_feature_importances": info["top_feature_importances"],
    }
