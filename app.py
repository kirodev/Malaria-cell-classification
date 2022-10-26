
import streamlit as st
import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras.preprocessing import image
#import cv2
import numpy as np
#import PIL
from PIL import Image, ImageOps



@st.cache(allow_output_mutation=True)

def load_model():
  model = tf.keras.models.load_model('efficientNetB0_mymodel1.hdf5')
  return model
with st.spinner('Model is being Loaded..'):
  model = load_model()

st.write(
    """
    Malaria Infected Cell Classification

    """
)

file = st.file_uploader("Please upload a cell image file", type=["jpg", "png"])

def predict(img, model):
    target_size = (224,224)
    img = ImageOps.fit(img, target_size)
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    img = img/255
    prob = model.predict(img)
    prediction = np.argmax(prob, axis=1)
     
     
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image)
    predictions = predict(image, model)
    if predictions == [1]:
        string = "Uninfected cell"
    else:
        string = "Parasitized cell"
    st.success(string)