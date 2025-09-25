import tensorflow as tf
from keras.utils import custom_object_scope

with custom_object_scope({'TrueDivide': tf.keras.layers.Lambda(lambda x: x)}):
    model = tf.keras.models.load_model("best_model.h5", compile=False)

model.save("best_model_tf", save_format="tf")
print("✅ Model başarıyla 'best_model_tf' klasörüne kaydedildi.")

