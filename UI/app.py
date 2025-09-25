import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

from keras.utils import custom_object_scope
import tensorflow as tf

@st.cache_resource
def load_model():
    with custom_object_scope({'TrueDivide': tf.keras.layers.Lambda(lambda x: x)}):
        model = tf.keras.models.load_model("best_model.h5", compile=False)
    return model


model = load_model()
class_names = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

st.title("🌍 Intel Image Classification – Akbank Bootcamp Project")

uploaded_file = st.file_uploader("Bir resim yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array)
    pred_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.success(f"Tahmin: **{pred_class}** ({confidence*100:.2f}%)")

