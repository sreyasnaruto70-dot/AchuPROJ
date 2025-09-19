import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('plant_cnn_model.h5')

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('plant_cnn_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted to TensorFlow Lite format.")