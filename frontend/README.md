# Frontend

Este directorio contiene solo el **esqueleto conceptual** para una futura interfaz web.

## Objetivo funcional

Permitir al usuario:

1. subir una señal ECG o una secuencia ya preprocesada
2. elegir la versión del modelo (`v1`, `v2`, `v3`)
3. lanzar inferencia
4. visualizar:
   - clase predicha
   - probabilidades
   - explicación temporal XAI

## Sugerencia de flujo

- página principal con selector de versión
- panel de carga de archivo o JSON
- tarjeta de resultado
- visualización temporal de relevancia por timestep / latido
