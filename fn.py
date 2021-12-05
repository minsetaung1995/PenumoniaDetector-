import keras
import numpy as np
import streamlit as st
from keras import layers, models, optimizers  # modeling
from PIL import Image

MODEL = "/content/gdrive/MyDrive/TeamKSM (2).h5"



def load_model():
    print("loading model")
    model = keras.models.load_model(f"{MODEL}", compile=True)

    return model


def preprocess_image(img):
    image = Image.open(img).convert("RGB")
    p_img = image.resize((150, 150))

    return np.array(p_img) / 255.0


def predict(model, img):
    prob = model.predict(np.reshape(img, [1, 150, 150, 3]))

    if prob > 0.5:
        prediction = True
    else:
        prediction = False

    return prob, prediction
 #App Title
st.title("Pneumodetector APP")

# Introduction text
st.markdown(unsafe_allow_html=True, body="<p>Welcome to Pneumodetector APP.</p>"
                                         "<p>This is a basic app built with Streamlit."
                                         "With this app, you can upload a Chest X-Ray image and predict if the patient "
                                         "from that image suffers pneumonia or not.</p>"
                                         "<p>The model used is a Convolutional Neural Network (CNN) and in this "
                                         "moment has a test accuracy of "
                                         "<strong>83.080.</strong></p>")

st.markdown("First, let's load an X-Ray Chest image.")

# Loading model

# Img uploader
img = st.file_uploader(label="Load X-Ray Chest image", type=['jpeg', 'jpg', 'png'], key="xray")

if img is not None:
    # Preprocessing Image
    p_img = preprocess_image(img)

    if st.checkbox('Zoom image'):
        image = np.array(Image.open(img))
        st.image(image, use_column_width=True)
    else:
        st.image(p_img)

    # Loading model
    loading_msg = st.empty()
    loading_msg.text("Predicting...")
    model = load_model()

    # Predicting result
    prob, prediction = predict(model, p_img)

    loading_msg.text('')

    if prediction:
        st.markdown(unsafe_allow_html=True, body="<span style='color:red; font-size: 50px'><strong><h4>Pneumonia! :slightly_frowning_face:</h4></strong></span>")
    else:
        st.markdown(unsafe_allow_html=True, body="<span style='color:green; font-size: 50px'><strong><h3>Healthy! :smile: </h3></strong></span>")

    st.text(f"*Probability of pneumonia is {round(prob[0][0] * 100, 2)}%")