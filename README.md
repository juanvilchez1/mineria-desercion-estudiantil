# Predicción de la deserción estudiantil

Proyecto de minería de datos con 4.424 registros del conjunto **Predict Students’ Dropout and Academic Success**.

## Objetivo
Comparar regresión logística, árbol de decisión y Random Forest para identificar estudiantes con riesgo de deserción.

## Metodología
- Variable objetivo binaria: `Desercion` (1 = Dropout, 0 = Enrolled/Graduate).
- División estratificada: 80 % entrenamiento y 20 % prueba.
- Validación cruzada estratificada de 5 particiones sobre entrenamiento.
- Métricas: exactitud, precisión, sensibilidad, F1 y ROC-AUC.
- Preprocesamiento dentro de pipelines para evitar fuga de información.

## Ejecución
```bash
pip install -r requirements.txt
python src/modelado.py
```

## Estructura
- `data/data.csv`: conjunto original.
- `src/modelado.py`: preprocesamiento, entrenamiento y validación.
- `results/`: tablas CSV.
- `figures/`: gráficos comparativos.

## Fuente
https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
