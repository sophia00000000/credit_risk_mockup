# Simulador ejecutivo de riesgo crediticio — Gradient Boosting

Aplicación web de apoyo académico basada en el informe del Parcial 2. El frontend llama a una API FastAPI y la API usa un Gradient Boosting real entrenado en Python. No hay una fórmula de riesgo escrita en JavaScript.

## Dataset

Se utiliza el **Credit Risk Dataset** de Kaggle: https://www.kaggle.com/datasets/laotse/credit-risk-dataset

Descarga `credit_risk_dataset.csv` y guárdalo en esta carpeta, al lado de `train_model.py`. El contenedor de esta entrega no pudo descargar directamente el binario desde Kaggle, por lo que no se incluye el CSV ni un modelo ficticio. La app muestra un mensaje claro hasta que entrenes el modelo.

## Ejecutar en Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Luego abre `index.html` en el navegador. La API queda en `http://127.0.0.1:8000`.

## Ejecutar en Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## Entrenamiento y decisiones

- Limpieza documentada: eliminar faltantes en `person_emp_length` y `loan_int_rate`, edades mayores a 100, antigüedad laboral mayor a 50 años e ingresos sobre el percentil 99.5.
- Partición estratificada 80/20 con `random_state=42`.
- Gradient Boosting: 300 estimadores, profundidad máxima 5, `learning_rate=0.1`, `min_samples_leaf=20`, `subsample=1.0`.
- Umbrales buscados con predicciones out-of-fold en entrenamiento, siguiendo los objetivos del notebook: mora de aprobados <=5% y precisión de la zona de rechazo >=90%. Se guardan en `model_info.json`; pueden no ser exactamente 0.200 y 0.360 si la versión del CSV o el preprocesamiento cambia.
- `model_info.json` contiene las métricas y las tasas observadas al ejecutar el entrenamiento. No se presentan como resultados nuevos antes de entrenar.

## Interpretación

- Verde: probabilidad por debajo del umbral de aprobación.
- Amarillo: zona intermedia; revisión manual obligatoria.
- Rojo: probabilidad igual o superior al umbral de rechazo, decisión propuesta sujeta a políticas del banco.

Es un prototipo académico. No debe usarse para tomar decisiones reales de crédito sin validación, pruebas de equidad, explicación individual, revisión legal y supervisión humana.
