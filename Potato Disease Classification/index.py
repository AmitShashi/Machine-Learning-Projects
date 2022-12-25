# Command to run :: streamlit run index.py

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

hide_streamlit_style = """ <style>
                                    # MainMenu {visibility: hidden;}
                                      footer {visibility: hidden;}
                           </style>
                       """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.write("""
    ## Intel Image Classification (TCS Internship)
    """)
st.write("### Upload an Image")

MODEL = tf.keras.models.load_model("./models/4")
CLASS_NAMES = class_names = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']


uploaded_file = st.file_uploader("Choose a file:")
print(uploaded_file)



if uploaded_file is not None:

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')
    with col2:
        st.image(uploaded_file)
    with col3:
        st.write(' ')
    image = Image.open(uploaded_file)

    image=image.resize([150,150])          # here we have resized the image using PIL

    image = np.array(image)

    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    print(predicted_class)
    print(confidence)

    st.title("Result : "); st.success(predicted_class)
    st.title("Accuracy : "); st.success(confidence * 100)
