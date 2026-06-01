# Predicción de cancelaciones hoteleras

Práctica de Evaluación Final del módulo de Machine Learning y Deep Learning

Sistema que entrena, evalúa y compara distintos modelos de clasificación
binaria sobre un dataset real de reservas hoteleras, selecciona el mejor según una
métrica principal y automatiza el flujo completo desde los datos crudos, 
exponiéndolo a través de una API REST.


## Descripción del problema y de los datos

El objetivo es predecir si una reserva de hotel será cancelada (`is_canceled = 1`)
o no (`is_canceled = 0`). Es un problema de **clasificación binaria**

- Dataset [data/dataset_practica_final.csv](data/dataset_practica_final.csv)
- Variable objetivo `is_canceled` (binaria).


### Observación importante

Las clases están relativamente desbalanceadas y el coste de una cancelación no detectada
no es simétrico al de una falsa alarma. Por ello se calculan un conjunto de métricas
(accuracy, precision, recall, F1 y ROC-AUC) y se usa F1 / ROC-AUC como criterio de
comparación, mas interesante debido a la naturaleza del problema.

En el problema propuesto, se asume que el coste de no detectar una reserva que se cancelará 
es mayor (habitación vacía) que predecir una cancelación que al final no se produce, pero 
no tan critico como para usar recall unicamente.


## Como ejecutar el proyecto

1. Crear el entorno e instalar dependencias:

   python -m venv env-modulo5
   source env-modulo5/bin/activate
   pip install -r requirements.txt

2. Arrancar la API y la web desde la raíz del proyecto:

   PYTHONPATH=src uvicorn api:app --reload --port 8000

   Abrir http://localhost:8000/ en el navegador. Desde la web se puede:
   - Entrenar un modelo (eligiendo modelo y variante).
   - Ver los modelos entrenados con sus métricas.
   - Cargar una reserva de ejemplo y predecir si se cancela.

   La variante `all` entrena los tres modelos seguidos.


## Estructura del proyecto

```
modulo5_evaluacion/
├── data/
│   ├── raw/dataset_practica_final.csv        # Datos crudos
│   ├── processed/dataset_preprocessed.parquet# Datos preprocesados (generados por el notebook)
│   └── dataset_practica_final.csv            # Copia del dataset original
├── notebooks/                                # EDA y experimentación
│   ├── evaluacion.ipynb                       # EDA + preprocesado
│   └── evaluacionModelos.ipynb                # Comparativa de modelos
├── src/
│   ├── data_loader_simple.py                  # Carga y preprocesado del CSV crudo
│   ├── model_trainer.py                       # Split, escalado, entrenamiento y evaluación
│   ├── keras_neural_network.py                # Modelo de red neuronal de Keras
│   └── api/                                   # API REST (FastAPI)
        ├── static                             # Web donde interactuar con la api
│       ├── main.py                            # Endpoints /train, /predict, /models, /sample
│       ├── model_store.py                     # Persistencia de modelos entrenados
│       └── predictor.py                       # Inferencia sobre filas crudas
├── models/                                    # Modelos entrenados (joblib + metadata.json)
├── requirements.txt                           # Dependencias
└── README.md
```

### Decisiones clave del preprocesado

El preprocesado (implementado en [src/data_loader_simple.py](src/data_loader_simple.py))
parte de los datos crudos y produce un dataset limpio sin nulos ni columnas de tipo objeto. 
Las decisiones principales:

**Columnas descartadas:**

- `reservation_status` y `reservation_status_date`: filtran información posterior a la
  reserva (data leakage) — conocerlas equivaldría a conocer ya la cancelación.
- `arrival_date_year`: sin sentido para predicciones presentes o futuras
- `agent` y `company`: con muchos nulos; se sustituyen por las banderas binarias
  `has_agent` y `has_company` (presencia/ausencia).
- `assigned_room_type`: redundante con `reserved_room_type` y solo conocida tras el
  check-in (hipotesis sin confirmación total)
- `country`: alta cardinalidad y escaso valor predictivo; se descarta. Además, Portugal
  es el valor por defecto al crearse la reserva según los autores.
- `distribution_channel` y `hotel`: la primera se descarta; de la segunda se deriva la
  bandera `is_resort_hotel`.
- columnas dudosas: `deposit_type` y `required_car_parking_spaces`: No está claro si 
  son variables fiables o data leakage. Se entrenan modelos con y sin ellas.

**Nuevas variables:**

- `total_guests = adults + children + babies` y
  `total_nights = stays_in_weekend_nights + stays_in_week_nights`.
- `adr_per_person` y `adr_per_night`: tarifa media normalizada por ocupantes y por
  noches.
- `arrival_month_sin` / `arrival_month_cos`: codificación cíclica del mes de llegada,
  para que diciembre y enero queden próximos (en lugar de un entero 1–12).
- Banderas binarias `has_agent`, `has_company`, `is_resort_hotel`.

**Limpieza e imputación:**

- `children` nulos → `0` (cast a entero).
- Categorías `"Undefined"` de `market_segment` y `distribution_channel` reasignadas a la
  categoría mayoritaria.
- Filtrado de filas inválidas: sin huéspedes o sin noches (`total_guests <= 0` o
  `total_nights <= 0`) y tarifas anómalas (`adr < 0` o `adr >= 5000`).
- Variables categóricas restantes (`market_segment`, `customer_type`,
  `reserved_room_type`, `meal`, `deposit_type`) mediante one-hot encoding
  (`drop_first=True`).

> En inferencia, `preprocess(df, drop_invalid_rows=False)` aplica las mismas
> transformaciones pero sin descartar filas de modo que cada registro recibe su
> predicción.


## Diseño del sistema

El flujo está separado en módulos reutilizables:

1. Carga y preprocesado — [src/data_loader_simple.py](src/data_loader_simple.py)
   - `load_and_preprocess(csv_path)`: lee el CSV crudo y devuelve el DataFrame procesado.
   - `preprocess(df, drop_invalid_rows=True)`: imputa datos, crea las
     variables nuevas (`total_guests`, `total_nights`, `adr_per_person`, codificación cíclica
     del mes…) y one-hot encoding. Con `drop_invalid_rows=False` se reutiliza en
     inferencia sin descartar filas.

2. Entrenamiento y evaluación — [src/model_trainer.py](src/model_trainer.py)
   - Tres variantes: `full`, `without_deposit_type`,
     `without_deposit_type_and_parking`. Debido a que estas variables pueden
	  influir altamente en el resultado y no sabemos si son válidas o no.
	  A modo de ejemplo, en un random forest el deposit_type es la variable con
	  mas influencia con un x3 respecto a la segunda:
	  	variable	importancia
		deposit_type_Non Refund	0.312276
		lead_time				0.108594
   - `train_test_split` estratificado (`test_size=0.2`, `random_state=42`).
   - Escalado de las variables continuas con `StandardScaler` (ajustado solo en
     train) cuando el modelo lo requiere.
   - `train(csv_path, model_name, variant)` orquesta todo el flujo y devuelve un
     `TrainingResult` con métricas, matriz de confusión, classification report,
     probabilidades y ranking de variables. Con `variant="all"` entrena las tres
     variantes a la vez.
   - Modelos disponibles en el pipeline:     
     `logistic_regression`, `decision_tree`, `random_forest`, `gradient_boosting`, `support_vector_machine`, `neural_network`
   La rama de trabajo de arboles incluye `decision_tree` y `gradient_boosting` en el pipeline ejecutable.
   Ambos modelos usan las tres variantes de variables y pueden entrenarse sin mover notebooks:
   `python -m src.train_tree_models --variant all`

   La comparativa exploratoria ampliada (árbol de decisión, boosting, red Keras…)
   se encuentra en los notebooks.

3. Persistencia e inferencia — [src/api/](src/api/)
   - `model_store.save/load/list_models`: guarda estimador, scaler y `metadata.json`
     bajo `models/<model_name>__<variant>__<timestamp>/`.
   - `predictor.predict`: reaplica el mismo preprocesado y escalado a filas crudas.

4. Web sencilla para interactuar con la api — [src/api/](src/api/)


## Anotaciones clave

1. LogisticRegression: Se utiliza un GridSearchCV para la búsqueda de mejores parametros
   (ver notebook evaluacionModelos) 

2. RandomForestClassifier: Se utiliza un RandomizedSearchCV para la búsqueda de mejores parametros
   (ver notebook randomForestnotebook)

3. DecisionTreeClassifier: Se integra como modelo interpretable de la rama de arboles. No requiere escalado y usa `class_weight="balanced"` como configuracion base.

4. GradientBoosting: Se integra como modelo boosting de la rama de arboles usando XGBoost (`XGBClassifier`)

5. SupportVectorMachine: Se utiliza `LinearSVC` con `class_weight="balanced"` y calibracion para exponer `predict_proba`.

## Resultados y conclusiones

Métricas sobre el conjunto de test, para los tres modelos y las tres variantes.

| Modelo | Variante | Accuracy | Recall (clase 1) | F1 (clase 1) | ROC-AUC |
|-|-|-|-|-|-|
| Regresión logística | full | 0.814 | 0.625 | 0.715 | 0.864 |
| Regresión logística | without_deposit_type | 0.796 | 0.650 | 0.704 | 0.847 |
| Regresión logística | without_deposit_type_and_parking | 0.791 | 0.628 | 0.692 | 0.833 |
| Random Forest | full | 0.866 | 0.758 | 0.808 | 0.931 |
| Random Forest | without_deposit_type | 0.866 | 0.759 | 0.808 | 0.931 |
| Random Forest | without_deposit_type_and_parking | 0.861 | 0.743 | 0.799 | 0.924 |
| Red neuronal (Keras) | full | 0.828 | 0.640 | 0.735 | 0.897 |
| Red neuronal (Keras) | without_deposit_type | 0.822 | 0.633 | 0.726 | 0.891 |
| Red neuronal (Keras) | without_deposit_type_and_parking | 0.812 | 0.595 | 0.702 | 0.876 |

Baseline (predecir siempre "no cancela"): 62.7%

Conclusiones:

- El Random Forest es el mejor modelo en todas las variantes (F1 ~0.81, ROC-AUC ~0.93)
  La red neuronal queda segunda (F1 0.735) y la regresión logística tercera (F1 0.70)
- Quitar `deposit_type` casi no cambia nada: en el Random Forest el F1 se mantiene igual
  (0.808) y en la logística incluso sube un poco el recall (0.603 → 0.650). El modelo no
  depende de esa variable sospechosa de leakage.
- Quitar además el parking baja poco en todos los modelos (Random Forest F1 0.799,
  red neuronal 0.702, logística 0.692). Por eso la variante conservadora, que solo usa
  variables cuyo mecanismo existiría en cualquier PMS, es defendible.
- El recall es el punto clave en todos los modelos: el  Random Forest detecta el 76% de
  las cancelaciones, la red neuronal el 64% y la logística entre el 60 y el 65%.
  En el análisis del problema se parte de la base que no detectar una cancelación es lo 
  más caro, por lo que se podría ajustar el umbral de decisión si esto es así. Por defecto
  se utiliza un umbral de 0.5
