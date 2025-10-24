import tensorflow as tf
import os

original_model_path = r"plant_disease_model.h5"
tflite_model_path = r"plant_disease_model.tflite"

# Check if file exists
if not os.path.exists(original_model_path):
    print(f"❌ File not found: {original_model_path}")
    exit()

# Load model
print("⏳ Loading model...")
model = tf.keras.models.load_model(original_model_path, compile=False)
print("✅ Model loaded successfully!")

# Convert to TensorFlow Lite with quantization
print("⏳ Converting to TensorFlow Lite with quantization...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enables quantization
tflite_model = converter.convert()

# Save the TF Lite model
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

# Check size
size_mb = os.path.getsize(tflite_model_path) / (1024*1024)
print(f"📦 Compressed TF Lite model size: {size_mb:.2f} MB")

if size_mb <= 25:
    print("🎉 Model is now under 25 MB!")
else:
    print("⚠️ Still larger than 25 MB. Consider using post-training integer quantization with a representative dataset.")
