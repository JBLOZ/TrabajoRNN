# CURRENT_PIPELINE_AUDIT.md

## Hallazgo central

El pipeline heredado presenta leakage estructural por:
- split aleatorio tras crear ventanas superpuestas
- validación dentro del conjunto ya sobremuestreado
- uso de SMOTE plano sobre secuencias
- posible ruptura de semántica temporal en los CSV derivados

## Consecuencia

Las métricas del notebook original son útiles como **señal exploratoria**, pero no como evidencia sólida de generalización inter-paciente.

## Decisión

Se rehace el proyecto con:
- splits grouped por registro
- entradas causales
- tratamiento prudente del desbalance
- comparación recurrente v1/v2/v3
- XAI temporal transversal
