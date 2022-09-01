# Aplicación de Técnicas de Machine Learning en la Clasificación de Noticias Falsas en el Idioma Castellano

*Resumen*

La presente investigación aborda el problema de la detección de noticias falsas en el idioma español o castellano utilizando técnicas de Text Mining y Machine Learning. Es importante recalcar que la problemática es la misma que se aborda para el idioma inglés, sin embargo, no existe una cantidad significativa de investigaciones y conjunto de datos debidamente etiquetados y estructurados en español para entrenar de manera efectiva un modelo de Machine Learning. Por lo tanto, esta investigación explora diferentes metodologías y estrategias para ser un referente en futuras investigaciones en esta problemática. Para la elaboración de la investigación se utilizó el nuevo corpus de noticias falsas en el idioma español, el cual recopila noticias de varias fuentes de distintos países hispano hablantes. Se utilizaron técnicas de text Mining para la explotación del texto, y se capturó la frecuencia del uso de las palabras en el idioma español o castellano con el CORPES de la RAE, así como las emociones de estas. Al aplicar las técnicas de Machine Learning se utilizó validación cruzada para determinar los mejores parámetros, así como validación cruzada repetida para obtener las tasas de fallos y precisión de los modelos. Según los resultados, tanto para una selección bajo el criterio AIC y BIC, el mejor modelo es una aplicación de Random Forest, con una precisión en las predicciones superior al 90% respecto al conjunto de datos. En general las frecuencias de las palabras y la fuente de las noticias resultaron variables significativas para el modelo, y el resto de las variables no son recomendadas para la construcción de futuros modelos.

Detalle de cada uno de los archivos:
- **DATOS.zip**: incluye los datos utilizados para el desarrollo de la investigación, incluye el corpus de noticias falsas, stopwords, y CORPES de la RAE.
- **import_data.R**: se importa los datos a utilizar y las fuentes de donde se extraen.
- **preproc.R**: Preprocesamiento a cada conjunto de datos.
- **main.R**: Transformación del texto de las noticias y generación de nuevas variables tales como TF-IDF, Análisis de Sentimiento, Frecuencias RAE, entre otras.
- **visual.R**: Análisis Exploratorio de los datos.
- **FUNCIONES.zip**: Funciones de Validación Cruzada Repetida para los modelos de Machine Learning.
- **models.R**: Creación de receta para la transformación de los datos para la construcción de los modelos de Machine Learning y Regresión Logística.

