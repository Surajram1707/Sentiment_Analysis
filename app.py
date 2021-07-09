import numpy as np
from fastapi import FastAPI, Form
import pandas as pd
from starlette.responses import HTMLResponse 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from model import preProcess_data
from pydantic import BaseModel
import tensorflow as tf
import re
def preProcess_data(text):
    
    text = text.lower()
    new_text = re.sub('[^a-zA-z0-9\s]','',text)
    new_text = re.sub('rt', '', new_text)
    return new_text

app = FastAPI()

data = pd.read_csv('Sentiment.csv')
tokenizer = Tokenizer(num_words=2000, split=' ')
tokenizer.fit_on_texts(data['text'].values)



def my_pipeline(text):
  text_new = preProcess_data(text)
  X = tokenizer.texts_to_sequences(pd.Series(text_new).values)
  X = pad_sequences(X, maxlen=28)
  return X


class inputToModel(BaseModel):
    text:str


@app.get('/')
def basic_view():
    return {"WELCOME": "GO TO /docs route, or /post or send post request to /predict "}



@app.get('/predict', response_class=HTMLResponse)
def take_inp():
    return '''<body style="background-color:pink;"><center><br><br><br><br><br><br><h1>Sentiment - Analysis</h1><br><form method="post"> 
    <input style="border:2px solid black;border-radius:20px;height:50px; width:400px; text-align:center;outline:none;background-color:lightpink;font-weight:bolder;" type="text" maxlength="28" name="text" placeholder="Text Emotion to be tested"/>  
    <input style="border:3px solid black;border-radius:20px;height:50px; width:200px; text-align:center; outline:none;background-color:violet;font-weight:bolder;" type="submit"/> 
    </form></center></body>'''



@app.post('/predict')
def predict(text:str = Form(...)):
    clean_text = my_pipeline(text)
    loaded_model = tf.keras.models.load_model('sentiment.h5')
    predictions = loaded_model.predict(clean_text)
    sentiment = int(np.argmax(predictions))
    probability = max(predictions.tolist()[0])*100
    print(sentiment)
    if sentiment==0:
        t_sentiment = 'negative'
    elif sentiment==1:
        t_sentiment = 'neutral'
    elif sentiment==2:
        t_sentiment='postive'
    
    return {
        "ACTUALL SENTENCE": text,
        "PREDICTED SENTIMENT": t_sentiment,
        "Probability Percentage": probability
    }
