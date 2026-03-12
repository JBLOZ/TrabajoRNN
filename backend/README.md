# Backend Flask

Este backend es un **esqueleto operativo**, no una aplicación final completa.

## Endpoints definidos

- `GET /health`
- `GET /models`
- `POST /predict`
- `POST /explain`

## Principios

- Nunca mezclar artefactos de preprocesado entre versiones.
- Cada versión carga su propio modelo y sus propios scalers.
- La respuesta debe devolver:
  - clase predicha
  - probabilidades por clase
  - metadatos de versión
  - explicaciones XAI cuando se pidan

## Entradas soportadas

### Modo recomendado ahora
`precomputed_sequence`

Permite enviar:
- `sequence_hrv`
- `sequence_fused`
- `sequence_hrv` + `sequence_morph`

### Modo futuro
`raw_signal`

El flujo está especificado, pero el backend actual no implementa todavía detección QRS completa; para ese modo se requiere proporcionar `r_peaks` o ampliar el preprocesado.

## Ejemplo de petición mínima

```json
{
  "model_version": "v1",
  "input_mode": "precomputed_sequence",
  "sequence_hrv": [[0.83, 0.0, 0.83, 0.0, 0.0, 0.0, 0.0, 0.0]]
}
```

## Ejecución

```bash
export MODEL_REGISTRY_PATH=backend/model_registry/registry.json
python backend/app.py
```
