# AGENTS.md

## Cómo deben trabajar futuros agentes sobre este repositorio

### 1. Antes de tocar modelos
Leer:
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/MODEL_VERSIONING.md`
- `docs/API_CONTRACT.md`

### 2. Prohibiciones
- No introducir modelos no recurrentes como solución final.
- No usar splits aleatorios por latidos.
- No aplicar SMOTE plano sobre secuencias para la solución principal.
- No usar `validation_split` si rompe la separación por grupos.
- No mezclar artefactos de preprocesado entre versiones.

### 3. Checklist antes de dar por buena una mejora
- ¿Se mantiene separación por registro/paciente?
- ¿El scaler se ajusta solo en train?
- ¿Val y test están libres de oversampling?
- ¿Se reporta macro-F1?
- ¿Hay métricas por clase?
- ¿Hay análisis por registro/paciente?
- ¿Hay XAI temporal?

### 4. Si una mejora sube mucho la accuracy
Asumir primero que puede existir leakage u optimismo metodológico.
