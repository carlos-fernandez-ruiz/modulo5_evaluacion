# Informe final: predicción de cancelaciones hoteleras

## 1. Definición de roles de la pareja

○ Regresión logística - Carlos
○ Árbol de decisión - Sheila
○ Random Forest - Carlos
○ Gradient Boosting (XGBoost, LightGBM o CatBoost) - Sheila
○ Red neuronal multicapa usando Keras de TensorFlow - Carlos
○ Support Vector Machine (SVM) - Sheila

## 2. Justificación del problema

El objetivo del proyecto es predecir si una reserva de hotel será cancelada (`is_canceled = 1`) o no (`is_canceled = 0`) a partir de la información disponible en el momento de la reserva. Se trata de un problema de clasificación binaria aplicado a un contexto real de negocio.

La predicción de cancelaciones es relevante porque permite anticipar habitaciones que probablemente quedarán vacías, ajustar estrategias de overbooking, mejorar la planificación de ingresos y optimizar recursos operativos. En este problema, no detectar una cancelación real puede tener un coste mayor que generar una falsa alarma, ya que una habitación vacía supone pérdida directa de ingresos. Aun así, el objetivo no es maximizar únicamente el recall, porque demasiadas falsas alarmas podrían producir decisiones comerciales poco eficientes.

Por este motivo, se evalúan varias métricas: accuracy, precision, recall, F1 y ROC-AUC. La métrica F1 se considera especialmente importante porque equilibra precision y recall, mientras que ROC-AUC permite medir la capacidad general del modelo para separar reservas canceladas y no canceladas.

## 3. Análisis exploratorio de datos

El dataset utilizado contiene 119.390 reservas y 32 columnas. La variable objetivo es `is_canceled`.

La distribución de la variable objetivo muestra un desbalance moderado:

| Clase | Significado | Registros | Porcentaje |
| --- | --- | ---: | ---: |
| 0 | Reserva no cancelada | 75.166 | 62,96% |
| 1 | Reserva cancelada | 44.224 | 37,04% |

Este desbalance no es extremo, pero sí suficiente para que accuracy no sea una métrica completa. Un modelo que predijera siempre "no cancelada" tendría una accuracy aproximada del 63%, pero no detectaría ninguna cancelación.

Durante el análisis exploratorio se identificaron varias cuestiones importantes:

| Variable | Observación | Decisión |
| --- | --- | --- |
| `company` | 112.593 valores nulos, aproximadamente 94,31% | No se usa directamente; se crea la variable binaria `has_company`. |
| `agent` | 16.340 valores nulos, aproximadamente 13,69% | No se usa directamente; se crea la variable binaria `has_agent`. |
| `country` | 488 valores nulos y 177 categorías distintas | Se descarta por alta cardinalidad y bajo valor práctico para el pipeline. |
| `children` | 4 valores nulos | Se imputan como 0. |
| `reservation_status` y `reservation_status_date` | Información posterior al resultado de la reserva | Se eliminan por riesgo claro de data leakage. |
| `assigned_room_type` | Se conoce después de la asignación final de habitación | Se elimina por posible información posterior a la reserva. |

También se detectaron filas inválidas o poco útiles para el entrenamiento:

- 180 reservas sin huéspedes.
- 715 reservas sin noches.
- 2 registros con tarifa `adr` anómala.

Tras el preprocesado, el dataset queda con 118.563 registros y 49 columnas en total: 48 variables predictoras más la variable objetivo. No quedan valores nulos ni columnas categóricas sin codificar.

### Decisiones de preprocesado

Las principales transformaciones aplicadas fueron:

- Eliminación de columnas con fuga de información: `reservation_status` y `reservation_status_date`.
- Eliminación de columnas poco fiables o redundantes: `arrival_date_year`, `assigned_room_type`, `country` y `distribution_channel`.
- Conversión de `agent` y `company` en variables binarias: `has_agent` y `has_company`.
- Creación de variables agregadas:
  - `total_guests = adults + children + babies`
  - `total_nights = stays_in_weekend_nights + stays_in_week_nights`
  - `adr_per_person`
  - `adr_per_night`
- Codificación cíclica del mes de llegada mediante `arrival_month_sin` y `arrival_month_cos`.
- One-hot encoding de variables categóricas como `market_segment`, `customer_type`, `reserved_room_type`, `meal` y `deposit_type`.
- Escalado de variables continuas con `StandardScaler` en los modelos que lo necesitan, como regresión logística, SVM y red neuronal.

Además, se definieron tres variantes de entrenamiento para estudiar el impacto de variables potencialmente problemáticas:

| Variante | Descripción |
| --- | --- |
| `full` | Usa todas las variables preprocesadas. |
| `without_deposit_type` | Elimina las columnas derivadas de `deposit_type`. |
| `without_deposit_type_and_parking` | Elimina `deposit_type` y `required_car_parking_spaces`. |

Estas variantes permiten comprobar si el modelo depende demasiado de variables que podrían no estar disponibles o podrían introducir sesgos en un escenario real.

## 4. Diseño del sistema

El sistema se diseñó como un flujo modular para separar carga de datos, preprocesado, entrenamiento, evaluación, persistencia e inferencia.

La estructura principal es:

```text
modulo5_evaluacion/
├── data/
│   └── dataset_practica_final.csv
├── notebooks/
│   ├── evaluacion.ipynb
│   ├── evaluacionModelos.ipynb
│   ├── logisticRegression.ipynb
│   ├── randomForestnotebook.ipynb
│   ├── neuralNetworknotebook.ipynb
│   ├── decision_tree_data_exploration.ipynb
│   ├── gradient_boosting_data_exploration.ipynb
│   └── support_vector_machine_data_exploration.ipynb
├── src/
│   ├── data_loader_simple.py
│   ├── model_trainer.py
│   ├── keras_neural_network.py
│   └── api/
│       ├── main.py
│       ├── model_store.py
│       ├── predictor.py
│       └── static/
├── models/
├── requirements.txt
└── README.md
```

El módulo `data_loader_simple.py` centraliza el preprocesado. Esto evita que el entrenamiento y la inferencia apliquen transformaciones diferentes. La función principal es `load_and_preprocess(csv_path)`, que carga el CSV y devuelve un dataframe numérico, limpio y listo para entrenar.

El módulo `model_trainer.py` contiene la lógica de entrenamiento y evaluación. Realiza un `train_test_split` estratificado con `test_size=0.2` y `random_state=42`, aplica escalado cuando el modelo lo necesita, entrena el estimador y devuelve métricas, matriz de confusión, reporte de clasificación y ranking de variables.

Los modelos contemplados en el pipeline son:

- Regresión logística.
- Árbol de decisión.
- Random Forest.
- Gradient Boosting con XGBoost.
- Support Vector Machine con `LinearSVC` calibrado.
- Red neuronal con Keras.

La API REST está implementada con FastAPI. Permite entrenar modelos, listar modelos guardados y hacer predicciones sobre reservas nuevas. Los modelos entrenados se guardan en `models/` junto con su metadata, lo que permite reutilizarlos sin tener que reentrenar siempre.

## 5. Resultados y elección final

Los modelos se evaluaron sobre el conjunto de test con las métricas accuracy, precision, recall, F1 y ROC-AUC. La siguiente tabla recoge las ejecuciones guardadas en `models/` para la variante `full` usadas en la comparación final.

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Regresión logística | 0,814 | 0,836 | 0,625 | 0,715 | 0,864 |
| Árbol de decisión | 0,801 | 0,718 | 0,767 | 0,742 | 0,884 |
| Gradient Boosting | 0,817 | 0,849 | 0,620 | 0,717 | 0,885 |
| Support Vector Machine | 0,814 | 0,812 | 0,651 | 0,723 | 0,864 |
| Random Forest | 0,862 | 0,841 | 0,776 | 0,807 | 0,929 |

El modelo con mejor rendimiento global es Random Forest. Obtiene el F1 más alto, 0,807, y también el mejor ROC-AUC, 0,929. Esto indica que no solo acierta más, sino que además ordena mejor las reservas según su probabilidad de cancelación.

El árbol de decisión es interesante por su interpretabilidad y obtiene un recall alto, 0,767, pero pierde precision y rendimiento global frente al Random Forest. Gradient Boosting ofrece una precision elevada, 0,849, pero detecta menos cancelaciones que Random Forest y que el árbol de decisión. SVM mejora ligeramente el F1 de la regresión logística, 0,723 frente a 0,715, pero mantiene un ROC-AUC similar y no alcanza el rendimiento de Random Forest. La regresión logística funciona como baseline interpretable, pero se queda por debajo en F1 y ROC-AUC.

Por tanto, la elección final del sistema es Random Forest, porque ofrece el mejor equilibrio entre detección de cancelaciones y control de falsas alarmas. Además, es robusto ante variables no escaladas, maneja bien relaciones no lineales y permite obtener importancias de variables para interpretar parcialmente el resultado.

Como criterio práctico, si el negocio quisiera priorizar todavía más la detección de cancelaciones, una mejora futura sería ajustar el umbral de decisión por debajo de 0,5 para aumentar recall, aceptando una posible bajada de precision.

## 6. Reflexión crítica sobre limitaciones y mejoras

Aunque el sistema consigue buenos resultados, tiene varias limitaciones importantes.

En primer lugar, algunas variables pueden no estar disponibles en todos los entornos reales o pueden depender del momento exacto en el que se hace la predicción. Por eso se analizaron variables como `deposit_type` y `required_car_parking_spaces` mediante variantes de entrenamiento. En un despliegue real sería necesario confirmar con expertos del dominio qué campos existen en el momento de la reserva y cuáles aparecen después.

En segundo lugar, el dataset corresponde a un contexto concreto de reservas hoteleras. El modelo podría no generalizar igual en otros países, cadenas hoteleras, temporadas o políticas de cancelación. Sería conveniente validar el sistema con datos más recientes y con datos de hoteles distintos.

En tercer lugar, se ha usado una separación train/test estratificada, pero no una validación temporal. En problemas de reservas, el tiempo puede ser importante porque cambian los hábitos de los clientes, los precios, la demanda y las políticas comerciales. Una mejora relevante sería evaluar el modelo con una partición temporal, entrenando con datos antiguos y probando con datos posteriores.

Otra limitación es que se evalúa principalmente con un umbral de clasificación estándar de 0,5. Sin embargo, la decisión de negocio podría requerir otro umbral. Si el coste de no detectar una cancelación es mucho mayor que el coste de una falsa alarma, convendría optimizar el umbral usando una matriz de costes.

También sería útil incorporar explicabilidad adicional, por ejemplo con SHAP, para entender mejor qué variables influyen en cada predicción individual. Esto ayudaría a justificar el uso del modelo ante usuarios no técnicos.

Como mejoras futuras se proponen:

- Validación temporal del modelo.
- Optimización del umbral de decisión según costes reales de negocio.
- Comparación completa de todas las variantes en todos los modelos.
- Registro de experimentos con una herramienta como MLflow.
- Monitorización de drift de datos y rendimiento tras el despliegue.
- Mejora de explicabilidad con SHAP o técnicas similares.
- Automatización de tests para asegurar que el preprocesado de entrenamiento e inferencia sigue siendo consistente.

## 7. Conclusión

El proyecto construye un sistema completo para predecir cancelaciones hoteleras desde datos crudos hasta una API REST utilizable. El análisis exploratorio permitió detectar valores nulos, variables con fuga de información, columnas de alta cardinalidad y registros inválidos. A partir de ello se diseñó un preprocesado reproducible y se compararon varios modelos de clasificación.

La elección final es Random Forest, ya que obtiene el mejor equilibrio entre F1, recall y ROC-AUC. El modelo es adecuado para el problema porque captura relaciones no lineales, mantiene buen rendimiento sin un preprocesado excesivamente complejo y permite interpretar la importancia relativa de las variables.

El sistema es funcional y defendible, aunque para un uso real sería necesario validar temporalmente el rendimiento, ajustar el umbral de decisión según costes de negocio y confirmar la disponibilidad real de las variables usadas en inferencia.
