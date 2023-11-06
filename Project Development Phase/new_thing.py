import keras
import tensorflow 
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.sequence import pad_sequences
from keras.applications import DenseNet201
from keras.layers import Activation, Dropout, Dense, Input, Layer
from keras.layers import Embedding, LSTM, add, Reshape, concatenate
from keras.models import Model, load_model
import numpy as np
import pickle

def feature_extraction(img_path):
    xtraction=load_model("extraction_model.h5")
    img=load_img(img_path,target_size=(224,224,3))
    img=img_to_array(img)
    img=np.expand_dims(img,axis=0)
    feature=xtraction.predict(img,verbose=0)
    return feature


model=load_model("overall_model.h5")

def index_to_word(integer,tokenizer):
    
    for word, index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None


def Predict_cap(model,image,tokenizer,max_length):
    in_text='startseq'
    for i in range(max_length):
        sequence=tokenizer.texts_to_sequences([in_text])[0]
        sequence=pad_sequences([sequence],max_length)
        yhat=model.predict([image,sequence],verbose=0)
        yhat=np.argmax(yhat)
        word=index_to_word(yhat,tokenizer)
        if word is None:
            break
        in_text+=" "+word
        
        if word=='endseq':
            break
    return in_text

with open ("tokenizer.pkl","rb") as file:
    tokenizer=pickle.load(file)