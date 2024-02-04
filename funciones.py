import pandas as pd
import fastapi
from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_steam = pd.read_csv('EDA/data_merged.csv')
data = data_steam.copy()
data_steam['Release_date'] = data_steam['Release_date'].astype(int)

# 1ra función: Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora. 
def developer(desarrollador: str):
    # Limpiar la cadena del desarrollador eliminando espacios adicionales y convirtiendo a minúsculas
    desarrollador = desarrollador.strip().lower()

    # Filtrar por el desarrollador proporcionado
    developer_data = data_steam[data_steam['Developer'].str.strip().str.lower() == desarrollador]

    # Agrupar por 'Release_date' y calcular las métricas
    result_data = developer_data.groupby('Release_date').agg({
        'Id_item': 'count',  # Cantidad total de ítems por año
        'Price': lambda x: (x == 0).sum(),  # Cantidad de ítems gratuitos por año
    }).reset_index()

    # Calcular el porcentaje de contenido gratis
    result_data['Porcentaje_contenido_gratis'] = round((result_data['Price'] / result_data['Id_item']) * 100, 2)

    # Seleccionar solo las columnas necesarias
    result_data = result_data[['Release_date', 'Id_item', 'Porcentaje_contenido_gratis']]

    # Renombrar las columnas
    result_data.columns = ['Release_date', 'Cantidad_items', 'Porcentaje_contenido_gratis']

    result_dict = {
        "Año": result_data['Release_date'].to_dict(),
        "Cantidad de items": result_data['Cantidad_items'].to_dict(),
        "Porcentaje_contenido_gratis": result_data['Porcentaje_contenido_gratis'].to_dict()
    }
    return result_dict


# 2da función: Debe devolver cantidad de dinero gastado por el usuario, el porcentaje de recomendación en base a reviews.recommend y cantidad de items.
def userdata(User_id: str):
    # Limpiar la cadena del usuario eliminando espacios adicionales y convirtiendo a minúsculas
    User_id = User_id.strip().lower()

    # Filtrar por el usuario proporcionado
    user_data = data_steam[data_steam['Id_user'].str.strip().str.lower() == User_id]

    # Agrupar por 'Id_user' y calcular las métricas
    result_data = user_data.groupby('Id_user').agg({
        'Recommend': 'sum',
        'Price': 'sum',  # Sumar el dinero gastado por el usuario
        'Id_item': 'count'  # Contar la cantidad de ítems comprados por el usuario
    }).reset_index()

    # Calcular el porcentaje de recomendaciones
    result_data['Porcentaje de recomendaciones'] = round((result_data['Recommend'] / result_data['Id_item']) * 100, 2)

    # Seleccionar solo las columnas necesarias
    result_data = result_data[['Id_user', 'Price', 'Porcentaje de recomendaciones','Id_item']]

    # Renombrar las columnas
    result_data.columns = ['Usuario', 'Dinero gastado', 'Porcentaje de recomendaciones','Id_item']

    # Obtener los valores como un diccionario
    user_dict = result_data.to_dict(orient='records')[0]

    # Formatear el resultado según el ejemplo
    formatted_result = {
        "Usuario": user_dict['Usuario'],
        "Dinero gastado": f"{user_dict['Dinero gastado']} USD",
        "Porcentaje de recomendaciones": f"{user_dict['Porcentaje de recomendaciones']}%",
        "Cantidad de ítems": user_dict['Id_item']
    }

    return formatted_result

# 3ra función Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.
def UserForGenre(genero: str):
    # Limpiar la cadena del género eliminando espacios adicionales y convirtiendo a minúsculas
    genero = genero.strip().lower()

    # Filtrar por el género proporcionado
    genre_data = data_steam[data_steam['Genres'].str.strip().str.lower() == genero]

    # Encontrar el usuario con más horas jugadas para el género dado
    user_entry = genre_data.loc[genre_data['Playtime_forever'].idxmax(), ['Id_user', 'Playtime_forever']]

    # Agrupar por 'Release_date' y calcular la acumulación de horas jugadas por año
    playtime_by_year = genre_data.groupby('Release_date')['Playtime_forever'].sum().reset_index()

    # Crear un diccionario para las horas jugadas por año
    hours_by_year = [{"Año": year, "Horas": hours} for year, hours in zip(playtime_by_year['Release_date'], playtime_by_year['Playtime_forever'])]

    # Crear el resultado final
    result = {
        "Usuario con más horas jugadas para Género": user_entry['Id_user'],
        "Horas jugadas": hours_by_year
    }

    return result

# 4ta función Devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos)

def best_developer_year(anio: int):
    data_1 = data_steam[['Release_date', 'Recommend', 'Developer', 'Sentiment_analysis']]
    
    # Filtrar los juegos por año y por recomendación positiva
    df_year = data_1[(data_1["Release_date"] == anio) & (data_1["Recommend"] == True) & (data_1["Sentiment_analysis"] == 2)]

    if df_year.empty:
        return {"error": "No hay datos para el año especificado."}

    # Contar el número de juegos recomendados por desarrollador y devolver los tres primeros desarrolladores
    top_desarrolladores = df_year["Developer"].value_counts().head(3).index.tolist()

    # Devolver el top 3 de desarrolladores
    return {"Puesto 1": top_desarrolladores[0], "Puesto 2": top_desarrolladores[1], "Puesto 3": top_desarrolladores[2]}




# 5ta función Según el desarrollador, se devuelve un diccionario con el nombre del desarrollador como llave y una lista con la cantidad total de registros de reseñas de usuarios 
# que se encuentren categorizados con un análisis de sentimiento como valor positivo o negativo.
def developer_reviews_analysis(desarrolladora: str):
    developer_data = data_steam[(data_steam['Developer'] == desarrolladora) &
                                data_steam['Sentiment_analysis'].isin([0, 2])]
    developer_data = developer_data.groupby(['Developer', 'Sentiment_analysis']).size().reset_index(name='count')

    # Crear un diccionario con el formato deseado
    result_dict = {desarrolladora: {'Negative': 0, 'Positive': 0}}

    for _, row in developer_data.iterrows():
        sentiment_label = 'Negative' if row['Sentiment_analysis'] == 0 else 'Positive'
        result_dict[desarrolladora][sentiment_label] = row['count']

    return result_dict

# 6ta función - Machine Learning Ingresando el id de producto, deberíamos recibir una lista con 5 juegos recomendados similares al ingresado.
def recomendacion_juego(Id_item):
    try:
        # Cargar datos
        data = pd.read_csv('Machine_Learning/steam_juegos.csv')
        data_steam_juegos = pd.read_csv('Machine_Learning/steam_id.csv')

        # Verificar si el Id_item existe en el conjunto de datos
        if Id_item not in data['Id_item'].values:
            raise ValueError(f"El Id_item {Id_item} no existe en el conjunto de datos.")

        # Crear matriz TF-IDF usando la columna Especificaciones
        matriz = TfidfVectorizer(min_df=1, max_df=0.8, token_pattern=r'(?u)\b\w\w+\b')
        vector = matriz.fit_transform(data['Especificaciones'])
        df_vector = pd.DataFrame(vector.toarray(), index=data['Id_item'], columns=matriz.get_feature_names_out())

        # Calcular similitud de coseno
        vector_similitud_coseno = cosine_similarity(df_vector.values)
        cos_similarity_df = pd.DataFrame(vector_similitud_coseno, index=df_vector.index, columns=df_vector.index)

        # Verificar si el Id_item tiene similitud con otros juegos
        if Id_item not in cos_similarity_df.index:
            raise ValueError(f"No hay juegos similares al Id_item {Id_item}.")

        # Obtener recomendaciones y realizar operaciones
        recomendacion_juego = cos_similarity_df.loc[Id_item]
        recomendacion = recomendacion_juego.sort_values(ascending=False)
        resultado = recomendacion.head(6).reset_index()
        df_resultado = resultado.merge(data_steam_juegos, on='Id_item', how='left')

        # Verificar si el Id_item existe en el conjunto de datos de steam_id
        if Id_item not in data_steam_juegos['Id_item'].values:
            raise ValueError(f"El Id_item {Id_item} no existe en el conjunto de datos de steam_id.")

        # Obtener nombre del juego
        nombre_juego = data_steam_juegos[data_steam_juegos['Id_item'] == Id_item]['App_name'].values[0]

        # Crear mensaje de recomendación
        texto_recomendacion = f"Debido a que te gusta el juego {Id_item} : {nombre_juego}, estoy seguro que te gustarán los siguientes juegos "
                
        # Crear y devolver el resultado
        result = {
            'mensaje': texto_recomendacion.strip(),
            'Recomendaciones de juegos': df_resultado['App_name'][1:6].tolist()
        }

        return result

    except Exception as e:
        return {'error': str(e)}
    
#7ma función Machine Learning se ingresa el Id_usuario y se le recomienda 5 juegos al mismo
def recomendacion_usuario(id_usuario):
    try:
        data_random_forest = data.copy()

        # Verifica si el usuario existe en el conjunto de datos
        if id_usuario not in data_random_forest['Id_user'].values:
            raise ValueError(f'El usuario {id_usuario} no existe en el conjunto de datos.')

        # Selecciona características (X) y la etiqueta (y)
        X = data_random_forest[['Id_item', 'Release_date', 'Price', 'Posted', 'Sentiment_analysis', 'Playtime_forever']]
        y = data_random_forest['Recommend']

        # Divide el conjunto de datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Crea y entrena el modelo
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X_train, y_train)

        # Filtra el conjunto de datos para obtener las características de juegos no etiquetados para el usuario
        juegos_sin_etiqueta = data_random_forest[data_random_forest['Id_user'] == id_usuario][['Id_item', 'Release_date', 'Price', 'Posted', 'Sentiment_analysis', 'Playtime_forever']]

        # Asegúrate de que haya al menos un juego sin etiquetar para el usuario
        if juegos_sin_etiqueta.empty:
            raise ValueError(f'No hay juegos sin etiquetar para el usuario {id_usuario}.')

        # Utiliza el modelo entrenado para predecir las preferencias del usuario para los juegos sin etiquetar
        preferencias_usuario = modelo.predict(juegos_sin_etiqueta)

        # Combina las predicciones con la información del juego y selecciona los 5 mejores
        juegos_sin_etiqueta['Recommend'] = preferencias_usuario
        juegos_recomendados = juegos_sin_etiqueta.sort_values(by='Recommend', ascending=False).head(5)

        # Realiza una fusión con el conjunto de datos original para obtener el nombre del juego
        juegos_recomendados = pd.merge(juegos_recomendados, data_random_forest[['Id_item', 'App_name']], on='Id_item', how='left')

        # Elimina duplicados basados en 'App_name'
        juegos_recomendados = juegos_recomendados.drop_duplicates(subset='App_name')

        # Reinicia el índice y luego incrementa en 1
        juegos_recomendados.reset_index(drop=True, inplace=True)
        juegos_recomendados.index += 1

        # Crear mensaje de recomendación
        mensaje_recomendacion = f'Recomendaciones para el usuario {id_usuario}:\n{juegos_recomendados.to_dict(orient="records")}'
        datos_dict = juegos_recomendados.to_dict(orient='records')
        return  datos_dict

    except Exception as e:
        return {'error': str(e)}