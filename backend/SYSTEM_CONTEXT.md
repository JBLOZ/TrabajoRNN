# SYSTEM_CONTEXT.md — Backend

## Rol

Servir inferencia y explicaciones de modelos recurrentes entrenados sobre MIT-BIH con un protocolo libre de leakage.

## Restricciones duras

- Nunca usar un artefacto de otra versión.
- Nunca rehacer escalado “a mano” durante inferencia.
- Nunca inferir con entradas cuya forma no coincida con `metadata.json`.
- Nunca describir la XAI como “pensamiento” de la red.
- Hablar de:
  - atribuciones
  - relevancia temporal
  - regiones influyentes
  - evidencia para la predicción

## Responsabilidades

1. Cargar `registry.json`.
2. Validar `model_version`.
3. Cargar:
   - `model.keras`
   - scalers necesarios
   - `metadata.json`
4. Aplicar exactamente el mismo preprocesado definido en entrenamiento.
5. Devolver JSON estable y versionado.

## Política de errores

- `400`: payload inválido
- `404`: versión no registrada
- `422`: forma de secuencia incompatible
- `500`: error interno de carga o predicción
