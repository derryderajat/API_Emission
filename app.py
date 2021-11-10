import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import pickle
import pandas as pd

df = pd.read_csv('./CO2 Emissions_Canada.csv')
df = df.drop_duplicates()
# Models
emission_vect = open("Model/emission.pkl", "rb")
emission_cv = pickle.load(emission_vect)

def findRealAttr(value, df=df):
    data = df.columns.to_list()
    string = value
    string.startswith(value)
    word = list(set([x if x.startswith(value) else "-" for x in data]))
    word.remove('-')
    return word[0]


# init app
app = FastAPI(debug=True)

origins = [
    "http://127.0.0.1:8000/"
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000/",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def sliceLink(value):
    value = value + '('
    index = list(value).index('(')
    return ''.join(list(value)[:index])
# Routes
@app.get('/')
async def index():
    return {"text":"Free API Emission"}

@app.get('/items/{name}')
async def get_items(name:str):
    return {"name":name }



@app.get('/data/scatter/{value}')
async def get_data(value:str):
    value = value.replace('%20', ' ')
    value = value.replace('%28', '(')
    value = value.replace('%29', ')')
    value = value.replace('%26', '&')
    x,y = value.split('&')
    x = findRealAttr(x)
    y = findRealAttr(y)
    data = df[[x,y]].copy()
    selected_columns = data.columns.tolist()
    selected_data = df[selected_columns].copy().rename({selected_columns[0]: 'x', selected_columns[-1]: 'y'}, axis=1).sample(n=(len(data)))
    json_data = selected_data.to_dict('records')
    return {'props':selected_columns,'data':json_data}


@app.get('/data/info/columns')
async def get_data_info_column():
    columns_name = list(df.select_dtypes(['float64', 'int64']).columns)
    return {'columns': columns_name}

# ML Aspect
@app.get('/predict/{value}')
async def predict(value):
    if (',' in value):
        value = value.split(',')
    if ('_' in value):
        value = value.split('_')
    
    prediction = emission_cv.predict(np.array([value]))[0]
    
    return {"orig_value":value, "prediction":prediction}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)   