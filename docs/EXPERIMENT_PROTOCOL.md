# EXPERIMENT_PROTOCOL.md

## 1. Problema

Predicción del latido objetivo a horizonte fijo (`horizon=1`) usando solo historia previa observada.

## 2. Unidad de muestra

Una muestra es una **secuencia histórica de latidos** perteneciente a un único registro, con etiqueta en el latido futuro objetivo.

## 3. Separación

### Protocolo principal
- split por `record_id`
- train / val / test sin solapamiento de grupos
- implementación recomendada:
  - `StratifiedGroupKFold` o estrategia equivalente con búsqueda de semilla

### Benchmark secundario
- split clásico inter-patient DS1/DS2 de de Chazal
- usar solo cuando la tarea y las clases lo permitan
- documentar si la clase `Q` queda ausente en train

## 4. Preprocesado

- scalers ajustados solo con train
- transformación posterior idéntica en val y test
- nada de leakage de estadísticas globales

## 5. Desbalance

Comparar:
- `class_weights`
- `focal loss`
- oversampling prudente por duplicación de secuencias reales
- `balanced batch sampler`

### Rechazo explícito
No usar SMOTE plano sobre secuencias aplanadas como solución principal porque:
- rompe coherencia temporal
- mezcla trayectorias fisiológicas incompatibles
- puede inflar validación si se valida dentro del pool sintético

## 6. Métricas obligatorias

- accuracy
- macro-F1
- weighted-F1
- precision/recall/F1 por clase
- ROC-AUC OVR por clase cuando sea aplicable
- PR-AUC por clase
- análisis por registro/paciente

## 7. XAI obligatoria

- saliency temporal
- integrated gradients
- occlusion temporal
- agregados por clase
- ejemplos correctos e incorrectos

## 8. Reproducibilidad

- seeds fijas
- config serializada
- checkpoints
- metadata JSON
- versiones v1/v2/v3 bien diferenciadas
