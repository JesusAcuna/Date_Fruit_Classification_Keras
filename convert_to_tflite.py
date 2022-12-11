import tensorflow as tf
from tensorflow import keras

model=keras.models.load_model('Best_Model_3.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('Best_Model_3.tflite','wb') as f_out:
    f_out.write(tflite_model)   