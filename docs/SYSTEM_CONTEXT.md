# SYSTEM_CONTEXT.md — Proyecto

## Objetivo del repositorio

Desarrollar y evaluar modelos **estrictamente recurrentes** para clasificación/predicción de arritmias sobre MIT-BIH con metodología rigurosa y reproducible.

## Reglas no negociables

1. El backbone final debe ser recurrente.
2. No usar CNN, Transformers, TCN o híbridos con backbone no recurrente.
3. Todo split debe respetar grupos de registro/paciente.
4. El test es intocable.
5. La XAI debe expresarse como atribución o relevancia, no como “pensamiento”.

## Meta principal

Un sistema defendible en un TFG serio, aunque la accuracy sea menor que la obtenida con protocolos contaminados.
