# MODEL_VERSIONING.md

## Versiones

### v1
- objetivo: baseline robusto
- entrada: HRV
- artefactos:
  - `models/v1/model.keras`
  - `models/v1/scaler_hrv.joblib`
  - `models/v1/metadata.json`

### v2
- objetivo: mejora técnica clara
- entrada: features fusionadas HRV + morfología
- artefactos:
  - `models/v2/model.keras`
  - `models/v2/scaler_fused.joblib`
  - `models/v2/metadata.json`

### v3
- objetivo: mejor sistema recurrente posible con coste razonable
- entrada: dos ramas, HRV y morfología
- artefactos:
  - `models/v3/model.keras`
  - `models/v3/scaler_hrv.joblib`
  - `models/v3/scaler_morph.joblib`
  - `models/v3/metadata.json`

## Metadata obligatoria

Cada versión debe guardar:
- nombre de versión
- seed
- dataset path
- split protocol
- class_names
- input shapes
- feature names
- balance strategy
- loss
- fecha de entrenamiento
- commit o hash del experimento si existe

## Regla
El backend nunca debe inferir si falta uno de los artefactos obligatorios para esa versión.
