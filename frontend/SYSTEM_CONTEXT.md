# SYSTEM_CONTEXT.md — Frontend

## Rol

Construir una interfaz simple y clara para inferencia sobre ECG con selección de versión de modelo.

## Restricciones

- No inventar preprocesado en cliente.
- No cambiar nombres de campos del contrato API.
- No interpretar atención como explicación suficiente.
- No ocultar qué versión del modelo produjo la predicción.
- Mostrar siempre:
  - versión
  - clase
  - probabilidades
  - explicación temporal

## Componentes sugeridos

- selector de modelo
- cargador de archivo / textarea JSON
- tabla de probabilidades
- heatmap temporal de atribuciones
- sección de advertencias del backend
