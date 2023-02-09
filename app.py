import streamlit as st
import pickle
from PIL import Image
from fastai.vision.all import *
from fastbook import search_images_bing
from pathlib import Path
from fastdownload import download_url

key= st.secrets["auth_key"]
path = Path()
learn_inf = load_learner(path/'model.pkl')


# get the name of the material from the user
material_name = st.text_input('Enter the name of the material')
# if the user enters a name
if material_name:
    # search for the material on bing
    results = search_images_bing(key=key, term=material_name)
    #download the first image
    im = download_url(results.attrgot('contentUrl')[0])
    # open the image
    im2 = PILImage.create(im)
    # show the image
    st.image(im2, caption='Your material', use_column_width=True)
    # run the image through the model
    pred,pred_idx,probs = learn_inf.predict(im)
    # show the result
    st.title(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')
