<<<<<<< HEAD
<p align=center><img src=https://d31uz8lwfmyn8g.cloudfront.net/Assets/logo-henry-white-lg.png><p>

# <h1 align=center> **PROYECTO INDIVIDUAL Nº1** </h1>

# <h1 align=center>**`Machine Learning Operations (MLOps)`**</h1>

En el proyecto realizado, se llevó a cabo el proceso completo del ciclo de vida de los datos, analizando datasets de la plataforma Steam con el fin de crear modelos de machine learning. Estos modelos, a su vez, proporcionan sistemas de recomendación que ofrecen una solución al problema planteado.

A lo largo del proyecto, se centró en alcanzar un Producto Mínimo Viable (MVP, por sus siglas en inglés) con el objetivo de obtener un dataset final. Este dataset se utiliza para la generación de consultas de los modelos de aprendizaje, que incluyen:

- El sistema de recomendación de juegos según el Id_item del juego.
- El sistema de recomendación de juegos ingresando el Id_usuario.

De esta manera, se logra abordar eficientemente el problema planteado.

De la misma manera, el proyecto tenía como segundo objetivo principal el de desarrollar una API para la realización de consultas por parte de la empresa, usando el framework FastApi. 

Enlace de GitHub: https://github.com/Chego01/Proyecto_individual_MlOps

Enlace de deployment: https://mlops-render-a5uk.onrender.com/docs

Enlace de video: 
<hr>  

## <h1 align=center> **`Extracción, Transformación y Carga de Datos`**


El proceso de ETL se desarrolló en la carpeta ETL, donde se encuentran los tres datasets tipo JSON. En este contexto, se trabajó utilizando la librería ast para leer línea por línea el archivo. Mediante la función literal_eval, cada línea se convertía en un diccionario, creando de esta manera un conjunto de diccionarios para, posteriormente, transformarlo en un dataframe.

A partir de aquí, se llevó a cabo un proceso ETL específico para cada dataset, adaptándolo a sus respectivas necesidades. A continuación, proporcionaré una explicación paso a paso del código:

1. `explode()`: Esta función se utiliza para desanidar los datos contenidos en la columna 'items'. Su función principal es transformar cada elemento de una lista en una fila independiente, al mismo tiempo que replica los valores de los índices existentes, generando una nueva columna por cada clave presente en los datos desanidados.

2. `json_normalize()`: Se utiliza para descomponer estructuras anidadas en datos JSON, como diccionarios dentro de diccionarios o listas dentro de diccionarios. La función transforma esta estructura anidada en un DataFrame, donde las claves anidadas se convierten en nombres de columnas, y los valores asociados se colocan en las celdas correspondientes.

3. `concat()`: Se utiliza para concatenar DataFrames a lo largo de un eje específico, ya sea a lo largo de las filas (eje 0) o a lo largo de las columnas (eje 1).

4.  `drop()`: Se utiliza para eliminar filas o columnas de un DataFrame. Puede ser aplicada tanto a filas como a columnas, y el DataFrame resultante no incluirá las filas o columnas eliminadas.

5. `dropna`: Se utiliza para eliminar filas que contienen valores nulos (NaN) en un DataFrame. Esta función proporciona una manera eficaz de manejar los datos faltantes.

6. `drop_duplicates`:Se utiliza para eliminar filas duplicadas en un DataFrame. Puede aplicarse para eliminar duplicados basándose en todas las columnas o en columnas específicas.

7. `to_csv`: Se utiliza para exportar un DataFrame a un archivo CSV (Comma-Separated Values). Este método proporciona una manera sencilla de guardar los datos en un formato tabular ampliamente utilizado.

8. `analyze_sentiment`: Esta función se realizo en el archivo `australian_user_reviews.ipynb` en donde se importo TextBlob  en la cual a partir de la columna Reviews se analizo si la recomendación era positiva,negativa o neutra asignando valores de 2,0,1 respectivamente.

**Unión de datasets**
Carpeta: ETL-Load_data
Documento: data_merged.ipynb

Para este paso los archivos .csv provenientes de los dataframes generados para cada archivo JSON se realizó previamente el proceso de ETL en donde se limpiaron los datos asi como también la transformación de los mismos como los nombres de las columnas, su orden dado

1. `data_merged = data_reviews.merge(data_games,on='Id_item',how='left')`:  En este proceso se realizo un primer merge entre dos datasets        (data_games,data_reviews) para crear un dataframe final mediante la unión de una columna en común la cual es Id_item.
2. `data = data_items.merge(data_merged,on='Id')`: Como segundo paso se realizo el merge final con el dataframe faltante realizandolo mediante la columna llamada Id.
3. `data = data.rename(columns={'Id_user_x':'Id_user','Id_item_x':'Id_item','Year_y':'Release_date','Year_x':'Posted'})`: Luego de esto se realiza el renombre de la columnas del dataframe final.
4. `data.drop('Id_user_y',axis='columns',inplace=True) - data.drop('Id_item_y',axis='columns',inplace=True)`: Para este paso se eliminan las columnas duplicadas que obtuvimos como resultante de los merges realizados
5. `data.to_csv('data_merged.csv',index=False)`: Como paso final se genera el csv del dataframe final llamado data_merged.csv.


## <h1 align=center> **`Análisis de Datos Exploratorio`** 
Carpeta: EDA
Documento: Eda.ipynb

Se generó un histograma para poder verificar los outliers presentes en cada columna del dataframe utilizado para lo cual se destaco en cuatro columnas: 'release_date’, ’price’ y ’playtime_forever'.
Como paso siguiente se cálculo los cuartiles y el rango interquartil (IQR) para posteriormente realizar un filtrado de los outliers en estas columnas.
# Calcular cuartiles e IQR para Release_date
`quartiles_release_date = data['Release_date'].quantile([0.25, 0.5, 0.75])`
`iqr_release_date = quartiles_release_date[0.75] - quartiles_release_date[0.25]`

# Calcular cuartiles e IQR para Price
`quartiles_price = data['Price'].quantile([0.25, 0.5, 0.75])`
`iqr_price = quartiles_price[0.75] - quartiles_price[0.25]`

# Calcular cuartiles e IQR para Playtime_forever
`quartiles_playtime_forever = data['Playtime_forever'].quantile([0.25, 0.5, 0.75])`
`iqr_playtime_forever = quartiles_playtime_forever[0.75] - quartiles_playtime_forever[0.25]`

Una vez realizado el cálculo del rango de datos en cada una de los columnas se crean los filtro o limites superiores e inferiores para eliminar los outliers. Una vez eliminados se observa un comportamiento normal de las variables tratadas.

# Relación de las variables 
Se procedio a analizar la relacion entre las variables del dataset final 
`scatter_matrix_all = sns.pairplot(filtered_data_all[columns_of_interest], palette='viridis')`: Mediante esta matriz se puede observar gráficamente la relación entre cada una de las variables, claro que esto se puede realizar de una forma numerica utilizando un heatmap o mapa de calor.

# Análisis de precios por el género de juegos 
Se analizó los precios de los juegos según el género de los mismo ya que, de esta forma nos puede dar una idea de los gustos de los usuarios mediante un boxplot. 
`top_genres = data['Genres'].value_counts().nlargest(6).index`

# Dispersión de datos en columnas como fecha lanzamiento,precios,tiempo juegado
Otro aspecto significativo es la variabilidad en los precios de los juegos a lo largo de los años y el tiempo de juego por parte de los usuarios. Esto proporciona una visión sobre la evolución de los juegos y las preferencias en términos de precios. Según el gráfico, se observa claramente una preferencia por los juegos lanzados a partir del año 2005, así como también por aquellos que tienen un valor menor.

`sns.scatterplot(x='Release_date', y='Playtime_forever', data=data, ax=axes[0])`
`sns.scatterplot(x='Release_date', y='Price', data=filtered_data_price, color='green', ax=axes[1])`

# Análisis del sentimiento 
Mediante este pairplot analizamos según el analisis de sentimiento de esta manera se da un acercamiento más profundo a la perspectiva del usuario 
`sns.pairplot(subset_data, hue='Sentiment_analysis', palette=custom_palette)`


# Visualización de datos
Para esta parte final se generaron otros gráficos en los cuales van más a detalle sobre las preferencias de los usuarios como: top 9 desarrolladores, top 9 géneros más preciados, proporción de recomendaciones positivas y negativas y el análisis de reseñas 

## <h1 align=center> **`Machine Learning`**
Carpeta: Machine_Learning
Documento: ML_funciones.ipynb

**Preparación de datos:** Para esto se trabajaron con tres datasets que son la copia del archivo data_merged.csv pero la razon de realizarles una copia es debido que se trabajo con columnas diferentes para el trabajo de los dos sistemas de recomendación.
`data.drop(['Id','Id_user','Genres','Posted','Sentiment_analysis','Recommend','Price','Playtime_forever'],axis=1,inplace=True)`

**Modelo de aprendizaje automático:**
**recomendacion_juego(Id_item):**
Para la función `def recomendacion_juego(Id_item)`: Se trabajo con las siguientes columnas Id_item, App_name, Developer,Release_date. En este sistema de recomendación se trabajo con el modelo `TfidfVectorizer` el cual convierte documentos de texto en vectores numéricos, donde cada dimensión del vector representa un término y el valor en esa dimensión representa la importancia del término en el documento. 
`matriz = TfidfVectorizer(min_df=1, max_df=0.8, token_pattern=r'(?u)\b\w\w+\b')`
`min_df=1`: Este parámetro establece el número mínimo de documentos en los que un término debe aparecer para ser incluido en la matriz TF-IDF. En este caso, se configura en 1, lo que significa que los términos deben aparecer en al menos un documento.

`max_df=0.8`: Este parámetro establece el umbral máximo de documentos en los que un término puede aparecer para ser incluido en la matriz TF-IDF. En este caso, se configura en 0.8, lo que significa que los términos que aparecen en más del 80% de los documentos serán excluidos.

`token_pattern`=r'(?u)\b\w\w+\b': Este parámetro define la expresión regular utilizada para extraer tokens (palabras) del texto. La expresión regular \b\w\w+\b coincide con palabras de al menos dos caracteres.

`vector = matriz.fit_transform(data['Especificaciones'])`:  Aplica el método fit_transform del objeto matriz para transformar el texto de la columna 'Especificaciones' en una matriz TF-IDF. El método fit_transform ajusta el vectorizador a los datos y transforma los datos de entrada en la representación TF-IDF.

`vector_similitud_coseno = cosine_similarity(df_vector.values)`:
Calcula la similitud coseno entre todos los pares de filas en la matriz de vectores df_vector. Después de ejecutar esta línea, `vector_similitud_coseno` contendrá una matriz que representa la similitud coseno entre las filas del conjunto de datos. Esto es útil para entender la similitud entre elementos en un espacio vectorial, como en tareas de recomendación.

`recomendacion_juego = cos_similarity_df.loc[Id_item]`: Selecciona la fila correspondiente al juego con el Id_item específico de la matriz de similitud coseno cos_similarity_df. Esta línea obtiene las similitudes coseno entre el juego dado (Id_item) y todos los demás juegos.

`recomendacion = recomendacion_juego.sort_values(ascending=False)`: Ordena los valores de similitud coseno en orden descendente, de modo que los juegos más similares al juego dado aparezcan primero en la serie recomendacion.

`resultado = recomendacion.head(6).reset_index()`: Selecciona los primeros 6 juegos más similares y los guarda en el DataFrame resultado. reset_index() reinicia los índices del DataFrame.

`df_resultado = resultado.merge(data_steam_juegos, on='Id_item', how='left')`: Combina la información de los juegos recomendados (resultado) con el DataFrame original data_steam_juegos utilizando la columna 'Id_item'. La combinación se realiza de manera izquierda (how='left'), asegurando que solo se mantengan las filas de resultado, pero ahora con la información adicional de data_steam_juegos.

**Modelo de aprendizaje automático:**
**recomendacion_usurario(Id_usuario):**

`X = data_random_forest[['Id_item', 'Release_date', 'Price', 'Posted', 'Sentiment_analysis', 'Playtime_forever']]`
`y = data_random_forest['Recommend']`: Selecciona características (X) y la etiqueta (y) escogiendo las columnas como variables independientes y la variable como Recommend como variable dependiente.
`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`:  En este punto, también se separa la cantidad de datos que se le dará para el entrenamiento del modelo RandomForest utilizado que es un clasificador de bosques aleatorios en scikit-learn. Un bosque aleatorio es un conjunto de árboles de decisión, y este clasificador crea múltiples árboles de decisión y los combina para obtener una predicción más robusta y generalizable..

`modelo = RandomForestClassifier(n_estimators=100, random_state=42)`
`n_estimators=100`: Especifica el número de árboles (estimadores) que se crearán en el bosque. En este caso, se han elegido 100 árboles.

`random_state=42`: Se utiliza para garantizar la reproducibilidad de los resultados. Fijar el random_state a un número específico asegura que, aunque el entrenamiento del modelo sea estocástico, los resultados serán los mismos en cada ejecución.

`modelo.fit(X_train, y_train)`: Esta línea de código entrena el modelo utilizando el conjunto de entrenamiento (X_train para características y y_train para etiquetas). Durante el entrenamiento, el modelo aprende a realizar predicciones basándose en las características proporcionadas y las etiquetas conocidas.

## <h1 align=center> **`Funciones y API`**
Documentos: funciones.py,funciones copy.ipynb, main.py
En los archivos funciones.py y funciones copy.ipynb se encuentran las funciones pero, en el caso de creacion de funciones copy.ipynb es para la verificación de su funcionamiento.
En el archivo main.py es donde se halla la creación de los endpoints que responden a solicitudes GET.

`/developer/{desarrollador}`: Este endpoint devuelve el número de elementos y el porcentaje de contenido gratuito por año para un desarrollador determinado.

`/userdata/{User_id}`: Este punto final devuelve la cantidad total gastada por un usuario, el porcentaje de recomendación basado en reseñas y la cantidad de artículos.

`/usergenre/{genero}`: Este endpoint devuelve el usuario que ha pasado más horas jugando para un género determinado y una lista de la acumulación de horas jugadas por año.

`/best_developer_year/{anio}`: Este endpoint devuelve los 3 principales desarrolladores con los juegos más recomendados por los usuarios para un año determinado.

`/developer_reviews_analysis/{desarrolladora}`: Este endpoint el número total de registros de reseñas de usuarios categorizados como sentimiento positivo o negativo.

`/recomendacion_juego/{Id_item}`: Este endpoint devuelve un mensaje en donde se reconoce segun el Id_item dado el nombre del juego y a su vez te recomienda un listado de 5 juegos

`/recomendacion_usuario/{id_usuario}`: Este endpoint retorna un diccionario en el cual al ingresar el Id_usuario que regresa un listado de 5 juegos recomendados para el usuario anidando información como el año de lanzamiento el analisis de sentimiento, nombre del juego y la existencia de recomendación dado 

Nota: Para hacer las consultas efectivas, se debe escribir en el campo respetando las mayusculas y minuscula.

Por ejemplo: en el caso de la primera función, se introduce el dato de esta manera, 'Valve', si se escribe 'valve' (con minuscula), no devolverá una respuesta a la consulta.

Contacto:

linkedin - linkedin.com/in/kevin-mayki-manchego-villegas-75b009213
email - kevmanchego@gmail.com

El estatus correspondiente al proyecto es de: completo/publicado.
=======
# <h1 align="center"> PROYECTO INDIVIDUAL Nº1 </h1>
# <h1 align="center">Kevin Manchego Villegas</h1>
# <h1 align="center">Machine Learning Operations (MLOps)</h1>

<p align="center">
    <img src="https://user-images.githubusercontent.com/67664604/217914153-1eb00e25-ac08-4dfa-aaf8-53c09038f082.png"  height=300>
</p>

En el proyecto realizado, se llevó a cabo el proceso completo del ciclo de vida de los datos, analizando datasets de la plataforma Steam con el fin de crear modelos de machine learning. Estos modelos, a su vez, proporcionan sistemas de recomendación que ofrecen una solución al problema planteado.

A lo largo del proyecto, se centró en alcanzar un Producto Mínimo Viable (MVP, por sus siglas en inglés) con el objetivo de obtener un dataset final. Este dataset se utiliza para la generación de consultas de los modelos de aprendizaje, que incluyen:

- El sistema de recomendación de juegos según el Id_item del juego.
- El sistema de recomendación de juegos ingresando el Id_usuario.

De esta manera, se logra abordar eficientemente el problema planteado.

De la misma manera, el proyecto tenía como segundo objetivo principal el de desarrollar una API para la realización de consultas por parte de la empresa, usando el framework FastApi. 

Enlace de GitHub: https://github.com/Chego01/Proyecto_individual_MlOps

Enlace de deployment:

Enlace de video: 
<hr>  


## **Descripción del problema (Contexto y rol a desarrollar)**

## Contexto

Tienes tu modelo de recomendación dando unas buenas métricas :smirk:, y ahora, cómo lo llevas al mundo real? :eyes:

El ciclo de vida de un proyecto de Machine Learning debe contemplar desde el tratamiento y recolección de los datos (Data Engineer stuff) hasta el entrenamiento y mantenimiento del modelo de ML según llegan nuevos datos.


## Rol a desarrollar

Empezaste a trabajar como **`Data Scientist`** en Steam, una plataforma multinacional de videojuegos. El mundo es bello y vas a crear tu primer modelo de ML que soluciona un problema de negocio: Steam pide que te encargues de crear un sistema de recomendación de videojuegos para usuarios. :worried:

Vas a sus datos y te das cuenta que la madurez de los mismos es poca (ok, es nula :sob: ): Datos anidados, de tipo raw, no hay procesos automatizados para la actualización de nuevos productos, entre otras cosas… haciendo tu trabajo imposible :weary: . 

Debes empezar desde 0, haciendo un trabajo rápido de **`Data Engineer`** y tener un **`MVP`** (_Minimum Viable Product_) para el cierre del proyecto! Tu cabeza va a explotar 🤯, pero al menos sabes cual es, conceptualmente, el camino que debes de seguir :exclamation:. Así que espantas los miedos y pones manos a la obra :muscle:

<p align="center">
<img src="https://github.com/HX-PRomero/PI_ML_OPS/raw/main/src/DiagramaConceptualDelFlujoDeProcesos.png"  height=500>
</p>

<sub> Nota que aquí se reflejan procesos, no herramientas tecnológicas. Haz el ejercicio de entender qué herramienta del stack corresponde a cada parte del proceso<sub/>

## **Propuesta de trabajo (requerimientos de aprobación)**

**`Transformaciones`**:  Para este MVP no se te pide transformaciones de datos(` aunque encuentres una motivo para hacerlo `) pero trabajaremos en leer el dataset con el formato correcto. Puedes eliminar las columnas que no necesitan para responder las consultas o preparar los modelos de aprendizaje automático, y de esa manera optimizar el rendimiento de la API y el entrenamiento del modelo.

**`Feature Engineering`**:  En el dataset *user_reviews* se incluyen reseñas de juegos hechos por distintos usuarios. Debes crear la columna ***'sentiment_analysis'*** aplicando análisis de sentimiento con NLP con la siguiente escala: debe tomar el valor '0' si es malo, '1' si es neutral y '2' si es positivo. Esta nueva columna debe reemplazar la de user_reviews.review para facilitar el trabajo de los modelos de machine learning y el análisis de datos. De no ser posible este análisis por estar ausente la reseña escrita, debe tomar el valor de `1`.

**`Desarrollo API`**:   Propones disponibilizar los datos de la empresa usando el framework ***FastAPI***. Las consultas que propones son las siguientes:

<sub> Debes crear las siguientes funciones para los endpoints que se consumirán en la API, recuerden que deben tener un decorador por cada una (@app.get(‘/’)).<sub/>


+ def **PlayTimeGenre( *`genero` : str* )**:
    Debe devolver `año` con mas horas jugadas para dicho género.
  
Ejemplo de retorno: {"Año de lanzamiento con más horas jugadas para Género X" : 2013}

+ def **UserForGenre( *`genero` : str* )**:
    Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.

Ejemplo de retorno: {"Usuario con más horas jugadas para Género X" : us213ndjss09sdf,
			     "Horas jugadas":[{Año: 2013, Horas: 203}, {Año: 2012, Horas: 100}, {Año: 2011, Horas: 23}]}

+ def **UsersRecommend( *`año` : int* )**:
   Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos/neutrales)
  
Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

+ def **UsersWorstDeveloper( *`año` : int* )**:
   Devuelve el top 3 de desarrolladoras con juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = False y comentarios negativos)
  
Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

+ def **sentiment_analysis( *`empresa desarrolladora` : str* )**:
    Según la empresa desarrolladora, se devuelve un diccionario con el nombre de la desarrolladora como llave y una lista con la cantidad total 
    de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor. 

Ejemplo de retorno: {'Valve' : [Negative = 182, Neutral = 120, Positive = 278]}

<br/>


> `Importante`<br>
El MVP _tiene_ que ser una API que pueda ser consumida segun los criterios de [API REST o RESTful](https://rockcontent.com/es/blog/api-rest/) desde cualquier dispositivo conectado a internet. Algunas herramientas como por ejemplo, Streamlit, si bien pueden brindar una interfaz de consulta, no cumplen con las condiciones para ser consideradas una API, sin workarounds.


**`Deployment`**: Conoces sobre [Render](https://render.com/docs/free#free-web-services) y tienes un [tutorial de Render](https://github.com/HX-FNegrete/render-fastapi-tutorial) que te hace la vida mas fácil :smile: . También podrías usar [Railway](https://railway.app/), o cualquier otro servicio que permita que la API pueda ser consumida desde la web.

<br/>

**`Análisis exploratorio de los datos`**: _(Exploratory Data Analysis-EDA)_

Ya los datos están limpios, ahora es tiempo de investigar las relaciones que hay entre las variables del dataset, ver si hay outliers o anomalías (que no tienen que ser errores necesariamente :eyes: ), y ver si hay algún patrón interesante que valga la pena explorar en un análisis posterior. Las nubes de palabras dan una buena idea de cuáles palabras son más frecuentes en los títulos, ¡podría ayudar al sistema de predicción! En esta ocasión vamos a pedirte que no uses librerías para hacer EDA automático ya que queremos que pongas en práctica los conceptos y tareas involucrados en el mismo. Puedes leer un poco más sobre EDA en [este articulo](https://medium.com/swlh/introduction-to-exploratory-data-analysis-eda-d83424e47151)

**`Modelo de aprendizaje automático`**: 

Una vez que toda la data es consumible por la API, está lista para consumir por los departamentos de Analytics y Machine Learning, y nuestro EDA nos permite entender bien los datos a los que tenemos acceso, es hora de entrenar nuestro modelo de machine learning para armar un **sistema de recomendación**. Para ello, te ofrecen dos propuestas de trabajo: En la primera, el modelo deberá tener una relación ítem-ítem, esto es se toma un item, en base a que tan similar esa ese ítem al resto, se recomiendan similares. Aquí el input es un juego y el output es una lista de juegos recomendados, para ello recomendamos aplicar la *similitud del coseno*. 
La otra propuesta para el sistema de recomendación debe aplicar el filtro user-item, esto es tomar un usuario, se encuentran usuarios similares y se recomiendan ítems que a esos usuarios similares les gustaron. En este caso el input es un usuario y el output es una lista de juegos que se le recomienda a ese usuario, en general se explican como “A usuarios que son similares a tí también les gustó…”. 
Deben crear al menos **uno** de los dos sistemas de recomendación (Si se atreven a tomar el desafío, para mostrar su capacidad al equipo, ¡pueden hacer ambos!). Tu líder pide que el modelo derive obligatoriamente en un GET/POST en la API símil al siguiente formato:

Si es un sistema de recomendación item-item:
+ def **recomendacion_juego( *`id de producto`* )**:
    Ingresando el id de producto, deberíamos recibir una lista con 5 juegos recomendados similares al ingresado.

Si es un sistema de recomendación user-item:
+ def **recomendacion_usuario( *`id de usuario`* )**:
    Ingresando el id de un usuario, deberíamos recibir una lista con 5 juegos recomendados para dicho usuario.


**`Video`**: Necesitas que al equipo le quede claro que tus herramientas funcionan realmente! Haces un video mostrando el resultado de las consultas propuestas y de tu modelo de ML entrenado! Recuerda presentarte, contar muy brevemente de que trata el proyecto y lo que vas a estar mostrando en el video.
Para grabarlo, puedes usar la herramienta Zoom, haciendo una videollamada y grabando la pantalla, aunque seguramente buscando, encuentres muchas formas más. 😉

<sub> **Spoiler**: El video NO DEBE durar mas de ***7 minutos*** y DEBE mostrar las consultas requeridas en funcionamiento desde la API y una breve explicación del modelo utilizado para el sistema de recomendación. En caso de que te sobre tiempo luego de grabarlo, puedes mostrar/explicar tu EDA, ETL e incluso cómo desarrollaste la API. <sub/>

<br/>

## **Criterios de evaluación y Rúbrica de Corrección**

**`Código`**: Prolijidad de código, uso de clases y/o funciones, en caso de ser necesario, código comentado. Se tendrá en cuenta el trato de los valores str como `COUNter-strike` / `COUNTER-STRIKE` / `counter-strike`.

**`Repositorio`**: Nombres de archivo adecuados, uso de carpetas para ordenar los archivos, README.md presentando el proyecto y el trabajo realizado. Recuerda que este último corresponde a la guía de tu proyecto, no importa que tan corto/largo sea siempre y cuando tu 'yo' + 1.5 AÑOS pueda entenderlo con facilidad. 

**`Cumplimiento`** de los requerimientos de aprobación indicados en el apartado `Propuesta de trabajo`

NOTA: Recuerde entregar el link de acceso al video. Puede alojarse en YouTube, Drive o cualquier plataforma de almacenamiento. **Verificar que sea de acceso público, recomendamos usar modo incógnito en tu navegador para confirmarlo**.

<br/>
Aquí te sintetizamos que es lo que consideramos un MVP aprobatorio, y la diferencia con un producto completo.



<p align="center">
<img src="https://github.com/HX-PRomero/PI_ML_OPS/raw/main/src/MVP_MLops.PNG"  height=250>
</p>

Acá van a poder encontrar una rúbrica, donde se específican los criterios de corrección utilizados:
- https://docs.google.com/spreadsheets/d/e/2PACX-1vR459kVWPsFGSBy6Hhzibp6hRVyvzSFUA0ta_v_FcMgNQnE84Kbt9XKIWLDPlJTqg/pubhtml?gid=1246267749&single=true 


## **Fuente de datos**

+ [Dataset](https://drive.google.com/drive/folders/1HqBG2-sUkz_R3h1dZU5F2uAzpRn7BSpj): Carpeta con el archivo que requieren ser procesados, tengan en cuenta que hay datos que estan anidados (un diccionario o una lista como valores en la fila).
+ [Diccionario de datos](https://docs.google.com/spreadsheets/d/1-t9HLzLHIGXvliq56UE_gMaWBVTPfrlTf2D9uAtLGrk/edit?usp=drive_link): Diccionario con algunas descripciones de las columnas disponibles en el dataset.
<br/>

## **Material de apoyo**

En este mismo repositorio podrás encontrar algunos (hay repositorios con distintos sistemas de recomendación) [links de ayuda](https://github.com/HX-PRomero/PI_ML_OPS/raw/main/Material%20de%20apoyo.md). Recuerda que no son los unicos recursos que puedes utilizar!

## **FAQ PI MLOps**
Les acercamos un notion donde tienen respuestas a algunas preguntas básicas de la etapa:
- https://soyhenry-data-labs.notion.site/PIMLops-FAQs-c7ca60c781e9468b8da4bd437c93412d 

  
<br/>
#   P r o y e c t o _ i n d i v i d u a l _ M l O p s 
 
 
>>>>>>> d679c3f76c340bbd3039b71850e1c61007291a73
