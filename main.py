import fastapi
import pandas as pd
from funciones import *
from fastapi import FastAPI
from typing import List,Dict,Tuple,Sequence,Callable, Optional,Any,Union
from typing import Union
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from fastapi import FastAPI, HTTPException

app = FastAPI()

data_steam = pd.read_csv('EDA/data_merged.csv')

@app.get("/")
#http://127.0.0.1:8000 Ruta madre del puerto

def root():
    return {'message':'Hello world'}

#Si funciona
@app.get("/developer/{desarrollador}")
async def desarrollador(desarrollador:str):
    try:
        resultado = developer(desarrollador)
        return resultado
    except Exception as e:

        return {'error': str(e)}  
#Si funciona
@app.get("/userdata/{User_id}")
def user(User_id:str):
    try:
        result = userdata(User_id)
        return result
    except Exception as e:
        return {'error': str(e)}  

#Si funciona
@app.get("/usergenre/{genero}")
def Genre(genero: str):
    try:
        result= UserForGenre(genero)
        return result
    except Exception as e:
        return {'error': str(e)}  
    
#No corre
@app.get("/bestdeveloperyear/{año}")
async def best_developer(año: int):
    try:
        
        result = best_developer_year(año)
        return result
    except Exception as e:
        return {'error': str(e)}


#Si corre
@app.get("/developer_reviews_analysis/{desarrolladora}")
def dev_reviews_analysis(desarrolladora: str):
    try:
        result= developer_reviews_analysis(desarrolladora)
        return result
    except Exception as e:
        return {'error': str(e)}  


#Si corre
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