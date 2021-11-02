import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import pickle



# Models
emission_vect = open("Model/emission.pkl", "rb")
emission_cv = pickle.load(emission_vect)


# init app
app = FastAPI(debug=True)

origins = [
    "https://flask-api-vehicle-emission",
    "http://127.0.0.1:8000/"
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
@app.get('/')
async def index():
    return {"text":"Hello API Master"}

@app.get('/items/{name}')
async def get_items(name:str):
    return {"name":name }


# ML Aspect
@app.get('/predict/{value}')
async def predict(value):
    # vect_value = emission_cv.transform([value]).toarray()
    # prediction = emission_cv.predict(vect_value)
    if (',' in value):
        value = value.split(',')
    if ('_' in value):
        value = value.split('_')
    
    prediction = emission_cv.predict(np.array([value]))[0]
    
    return {"orig_value":value, "prediction":prediction}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
