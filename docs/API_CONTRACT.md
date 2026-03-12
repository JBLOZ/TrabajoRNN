# API_CONTRACT.md

## GET /health

### Response
```json
{
  "status": "ok",
  "service": "mitbih-recurrent-api"
}
```

## GET /models

### Response
Devuelve el contenido del registro de modelos con:
- nombre
- descripción
- inputs esperados
- métodos XAI
- directorio de artefactos

## POST /predict

### Request
```json
{
  "model_version": "v1",
  "input_mode": "precomputed_sequence",
  "sequence_hrv": [[0.81, 0.0, 0.81, 0.0, 0.0, 0.0, 0.0, 0.0]]
}
```

### Opciones de payload

#### v1
- `sequence_hrv`

#### v2
- `sequence_fused`

#### v3
- `sequence_hrv`
- `sequence_morph`

#### futuro raw
- `signal`
- `sampling_rate_hz`
- `r_peaks` o detector QRS backend

### Response
```json
{
  "model_version": "v1",
  "predicted_class_index": 2,
  "predicted_class_name": "VEB",
  "probabilities": {
    "Normal (N)": 0.12,
    "SVEB": 0.08,
    "VEB": 0.71,
    "Fusión (F)": 0.03,
    "Desconocido (Q)": 0.06
  },
  "task": "next_beat_aami5_grouped",
  "warnings": []
}
```

## POST /explain

### Request
Mismo esquema que `/predict`.

### Response
```json
{
  "prediction": { "...": "igual que /predict" },
  "xai": {
    "saliency": "...",
    "integrated_gradients": "...",
    "occlusion": "..."
  }
}
```

## Validaciones mínimas

- `model_version` obligatorio
- secuencias obligatorias según versión
- forma temporal compatible con la metadata del modelo
- tipos numéricos válidos
