import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# Eğitilmiş bir derin öğrenme modeli kullanın
model = keras.applications.ResNet50(weights='imagenet')

# Bir resim üzerinde tahmin yapmak için bir fonksiyon oluşturun
def predict_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    results = decode_predictions(predictions, top=3)[0]

    for result in results:
        print(f'{result[1]}: {result[2]*100:.2f}%')

# Bir resim üzerinde tahmin yapın
image_path = r'C:\Users\ardak\Desktop\machine learning\koltuk.jpg'
predict_image(image_path)
