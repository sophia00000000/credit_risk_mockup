"""Entrena Gradient Boosting con la preparación descrita en el notebook del parcial."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, brier_score_loss,
                             confusion_matrix)
from sklearn.inspection import permutation_importance
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "credit_risk_dataset.csv"
MODEL_PATH = BASE_DIR / "gbm_bundle.joblib"
INFO_PATH = BASE_DIR / "model_info.json"
RANDOM_STATE = 42

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "No se encontró credit_risk_dataset.csv. Descárgalo de Kaggle y guárdalo "
        "junto a train_model.py. Consulta README.md para el enlace e instrucciones."
    )

# Misma limpieza documentada en el notebook.
df = pd.read_csv(DATA_PATH)
required = {
    "person_age", "person_income", "person_home_ownership", "person_emp_length",
    "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate", "loan_status",
    "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length"
}
missing = sorted(required - set(df.columns))
if missing:
    raise ValueError(f"Al CSV le faltan columnas esperadas: {missing}")

d = df.dropna(subset=["person_emp_length", "loan_int_rate"]).copy()
d = d[d["person_age"] <= 100]
d = d[d["person_emp_length"] <= 50]
d = d[d["person_income"] <= d["person_income"].quantile(0.995)].reset_index(drop=True)
# Evitar fuga accidental: loan_status es únicamente el objetivo.
y = d["loan_status"].astype(int)
X = pd.get_dummies(d.drop(columns=["loan_status"])).astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Hiperparámetros reportados en el notebook del parcial.
model = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=20,
    subsample=1.0,
    random_state=RANDOM_STATE,
)
model.fit(X_train, y_train)
p_test = model.predict_proba(X_test)[:, 1]

# Umbrales operativos: buscar en predicciones out-of-fold, sin usar test para fijarlos.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
p_oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=None)[:, 1]
grid = np.round(np.arange(0.01, 0.99, 0.005), 3)
valid_low = [t for t in grid if np.sum(p_oof < t) > 0 and y_train[p_oof < t].mean() <= 0.05]
valid_high = [t for t in grid if np.sum(p_oof >= t) > 0 and y_train[p_oof >= t].mean() >= 0.90]
# Fallbacks only if the new download/version does not admit a threshold satisfying the notebook's goals.
t_low = float(max(valid_low)) if valid_low else 0.20
t_high = float(min(valid_high)) if valid_high else 0.36
if t_high <= t_low:
    t_low, t_high = 0.20, 0.36

pred05 = (p_test >= 0.5).astype(int)
cm = confusion_matrix(y_test, pred05, labels=[0, 1])
metrics = {
    "accuracy": float(accuracy_score(y_test, pred05)),
    "precision": float(precision_score(y_test, pred05, zero_division=0)),
    "recall": float(recall_score(y_test, pred05, zero_division=0)),
    "f1": float(f1_score(y_test, pred05, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, p_test)),
    "pr_auc": float(average_precision_score(y_test, p_test)),
    "brier": float(brier_score_loss(y_test, p_test)),
    "test_rows": int(len(y_test)),
    "clean_rows": int(len(d)),
    "default_rate_dataset": float(y.mean()),
    "confusion_matrix_labels_0_1": cm.tolist(),
}

def zone_summary(prob, actual):
    prob = np.asarray(prob)
    actual = np.asarray(actual)
    zones = np.where(prob < t_low, "approval", np.where(prob >= t_high, "reject", "review"))
    out = {}
    for key, label in [("approval", "Aprobación automática"), ("review", "Revisión manual"), ("reject", "Rechazo automático propuesto")]:
        mask = zones == key
        out[key] = {
            "label": label,
            "count": int(mask.sum()),
            "share": float(mask.mean()) if len(mask) else 0.0,
            "observed_default_rate": float(actual[mask].mean()) if mask.any() else None,
        }
    return out

zones_test = zone_summary(p_test, y_test)
# Importancia por permutación, siguiendo el enfoque del notebook; una repetición mantiene el entrenamiento razonable.
pi = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=3,
                            random_state=RANDOM_STATE, n_jobs=1)
importance = sorted(
    [{"feature": str(name), "importance": float(value)} for name, value in zip(X.columns, pi.importances_mean)],
    key=lambda row: row["importance"], reverse=True
)[:12]

bundle = {"model": model, "columns": list(X.columns), "t_low": t_low, "t_high": t_high}
joblib.dump(bundle, MODEL_PATH)
info = {
    "model_name": "Gradient Boosting (GBM)",
    "parameters": {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.1, "min_samples_leaf": 20, "subsample": 1.0, "random_state": RANDOM_STATE},
    "thresholds": {"approval_below": t_low, "reject_at_or_above": t_high, "manual_review_between": True},
    "metrics": metrics,
    "zones_test": zones_test,
    "top_feature_importances": importance,
    "note": "Métricas generadas al ejecutar este entrenamiento con el CSV disponible; pueden variar si cambia la versión o preparación del dataset. Las tasas por zona son observaciones del conjunto de prueba, no garantías futuras."
}
INFO_PATH.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Modelo guardado: {MODEL_PATH}")
print(f"Metadatos guardados: {INFO_PATH}")
print(f"Umbral de aprobación < {t_low:.3f}; umbral de rechazo >= {t_high:.3f}")
print("Métricas de prueba:", json.dumps(metrics, ensure_ascii=False, indent=2))
print("Zonas en prueba:", json.dumps(zones_test, ensure_ascii=False, indent=2))
