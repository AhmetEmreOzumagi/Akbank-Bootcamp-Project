import streamlit as st
import tensorflow as tf
import numpy as np
from keras.utils import custom_object_scope
from PIL import Image

# ✅ Model yükleme fonksiyonu
@st.cache_resource
def load_model():
    # "TrueDivide" hatasını custom scope ile çözüyoruz
    with custom_object_scope({'TrueDivide': tf.keras.layers.Lambda(lambda x: x)}):
        model = tf.keras.models.load_model("best_model.h5", compile=False)
    return model

# ✅ Görseli işleme fonksiyonu
def preprocess_image(image: Image.Image):
    img = image.resize((224, 224))   # modelin input boyutu
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# ✅ Streamlit arayüzü
def main():
    st.title("🔬 Akbank Bootcamp - Görüntü Sınıflandırma Demo")

    uploaded_file = st.file_uploader("Bir resim yükleyin", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Yüklenen Görsel", use_column_width=True)

        if st.button("Tahmin Et"):
            with st.spinner("Model çalışıyor..."):
                model = load_model()
                img_array = preprocess_image(image)
                prediction = model.predict(img_array)

                st.subheader("📌 Sonuç")
                st.write(f"Tahmin: **{np.argmax(prediction)}**")
                st.bar_chart(prediction[0])

if __name__ == "__main__":
    main()
