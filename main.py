import fastapi
import pandas as pd
from funciones import *
from fastapi import FastAPI
from typing import List,Dict,Tuple,Sequence,Callable, Optional,Any,Union
from typing import Union
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
import uvicorn

app = FastAPI()

#data_steam = pd.read_csv('EDA/data_merged.csv')

@app.get("/")
#http://127.0.0.1:8000 Ruta madre del puerto

def root():
    return {'message':'Hello world'}

# 1ra función: Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora. 
@app.get("/developer/{desarrollador}")
async def desarrollador(desarrollador:str):
    try:
        resultado = developer(desarrollador)
        return resultado
    except Exception as e:

        return {'error': str(e)}  
    
# 2da función: Debe devolver cantidad de dinero gastado por el usuario, el porcentaje de recomendación en base a reviews.recommend y cantidad de items.
@app.get("/userdata/{User_id}")
def user(User_id:str):
    try:
        result = userdata(User_id)
        return result
    except Exception as e:
        return {'error': str(e)}  

# 3ra función Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.
@app.get("/usergenre/{genero}")
def Genre(genero: str):
    try:
        result= UserForGenre(genero)
        return result
    except Exception as e:
        return {'error': str(e)}  
    
# 4ta función Devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos)
@app.get("/best_developer_year/{anio}")
async def best_developer(anio: Union[int, str]):
    try:
        year = int(anio)
        result = best_developer_year(year)
        return result
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


# 5ta función Según el desarrollador, se devuelve un diccionario con el nombre del desarrollador como llave y una lista con la cantidad total de registros de reseñas de usuarios 
# que se encuentren categorizados con un análisis de sentimiento como valor positivo o negativo.
@app.get("/developer_reviews_analysis/{desarrolladora}")
def dev_reviews_analysis(desarrolladora: str):
    try:
        result= developer_reviews_analysis(desarrolladora)
        return result
    except Exception as e:
        return {'error': str(e)}  


# 6ta función - Machine Learning Ingresando el id de producto, deberíamos recibir una lista con 5 juegos recomendados similares al ingresado.
@app.get("/recomendacion_juego/{Id_item}")
def get_recomedacion_juego(Id_item: int):
    try:
        result= recomendacion_juego((Id_item))
        return result
    except Exception as e:
        return {'error': str(e)}  


@app.get("/recomendacion_usuario/{id_usuario}")
def get_recomedacion_usuario (id_usuario: str):
    try:
        result= recomendacion_usuario(id_usuario)
        return result
    except Exception as e:
        return {'error': str(e)}  

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)