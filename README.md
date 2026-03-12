# MIT-BIH Recurrent TFG Project

Repositorio de reconstrucción metodológica para un TFG de predicción de arritmias con **arquitecturas exclusivamente recurrentes** sobre MIT-BIH Arrhythmia Database.

## Objetivo

Rehacer el pipeline actual con foco en:

- evaluación inter-paciente o inter-registro
- prevención explícita de data leakage
- modelos recurrentes defendibles en un TFG serio
- interpretabilidad temporal real
- preparación ligera para despliegue con Flask

## Estructura

- `notebooks/`: notebooks v1, v2, v3 y comparativo
- `src/`: código modular reutilizable
- `models/`: artefactos exportados por versión
- `artifacts/`: métricas, gráficos y salidas auxiliares
- `backend/`: esqueleto Flask para inferencia y explicación
- `frontend/`: contexto y estructura mínima para futura interfaz web
- `docs/`: contratos, protocolo experimental y documentación para agentes

## Dataset esperado

Coloca MIT-BIH en:

```text
data/raw/mit-bih-arrhythmia-database-1.0.0/
```

Se esperan archivos `.dat`, `.hea` y `.atr` por registro.

## Principios metodológicos

1. **Nada de split aleatorio por latidos o ventanas**.
2. **Nada de `validation_split` sobre tensores ya mezclados o sobremuestreados**.
3. **Nada de SMOTE plano sobre secuencias como estrategia principal**.
4. **Scalers ajustados solo con train**.
5. **Test intocable**.
6. **El backbone final siempre es recurrente**.

## Arranque rápido

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Luego abre los notebooks en orden:

1. `01_v1_baseline_recurrent.ipynb`
2. `02_v2_improved_recurrent.ipynb`
3. `03_v3_best_recurrent_system.ipynb`
4. `04_comparative_analysis.ipynb`

## Nota importante sobre DS1/DS2

El split inter-patient clásico de de Chazal es útil para comparabilidad con literatura, pero en la práctica del problema **AAMI 5 clases** introduce una limitación: la clase `Q` puede quedar ausente en train según la partición clásica reproducida por múltiples repositorios. Por eso el repositorio usa como protocolo principal un split **grouped + stratified** a nivel de registro, y deja DS1/DS2 como benchmark secundario y documentado en `docs/EXPERIMENT_PROTOCOL.md`.
