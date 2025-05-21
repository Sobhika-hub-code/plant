# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
import json
import os
import matplotlib.pyplot as plt

# Load disease information from JSON file
json_path = "D:/Individual Project39/data/disease_info.json"  # Ensure this file exists
with open(json_path, 'r') as f:
    disease_info = json.load(f)

# Define dataset path
DATASET_PATH = "D:/Individual Project39/PlantVillage/"

# Image settings
IMG_SIZE = (224, 224)  # Resize all images
BATCH_SIZE = 32        # Number of images per batch
EPOCHS = 20            # Training epochs

# Data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=20, width_shift_range=0.2,
    height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, validation_split=0.2  # 80% train, 20% validation
)

# Load Train Data
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", subset="training"
)

# Load Validation Data
val_generator = train_datagen.flow_from_directory(
    DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", subset="validation"
)

# Get number of disease classes
NUM_CLASSES = train_generator.num_classes
print(f"🔹 Total Classes Found: {NUM_CLASSES}")

# Build the Model using MobileNetV2
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # Freeze base model layers

# Add Custom Layers
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

# Create Final Model
model = Model(inputs=base_model.input, outputs=output)

# Compile the Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# Save the Trained Model
MODEL_PATH = "D:\Individual Project39\models\plant_disease_model.h5"
model.save(MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")

# Plot Accuracy and Loss Curves
plt.figure(figsize=(12, 4))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Model Loss")

plt.show()
