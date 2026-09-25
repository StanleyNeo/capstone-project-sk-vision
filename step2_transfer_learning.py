import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import f1_score

print("Loading CIFAR-10 data...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 1. UPSCALE: Resize images from 32x32 to 96x96
print("Resizing images to 96x96 (MobileNetV2 minimum)...")
x_train = tf.image.resize(x_train, (96, 96))
x_test = tf.image.resize(x_test, (96, 96))

# 2. PREPROCESS: MobileNetV2 expects values in [-1, 1]
x_train = tf.keras.applications.mobilenet_v2.preprocess_input(x_train)
x_test = tf.keras.applications.mobilenet_v2.preprocess_input(x_test)

print("\nBuilding MobileNetV2 Transfer Learning model...")
# Load base model with correct input shape
base_model = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')

# 3. FINE-TUNE: Unfreeze the top 30 layers
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Build the new model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# 4. LOWER LEARNING RATE: Crucial for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\nStarting transfer learning training (5 epochs)...")
start_train = time.time()
# Increased batch size to handle larger images efficiently
model.fit(x_train, y_train, epochs=5, validation_split=0.1, batch_size=128)
end_train = time.time()

print("\nEvaluating on test set...")
start_pred = time.time()
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
end_pred = time.time()

# Calculate metrics
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
f1 = f1_score(y_test, y_pred, average='weighted')

# Save the model
model.save('mobilenetv2.keras')
model_size = os.path.getsize('mobilenetv2.keras') / (1024 * 1024) # MB

print("\n" + "="*50)
print(" TRANSFER LEARNING METRICS (MobileNetV2)")
print("="*50)
print(f" Accuracy:       {accuracy:.4f}")
print(f" F1 Score:       {f1:.4f}")
print(f" Train Time:     {end_train - start_train:.2f} seconds")
print(f" Predict Time:   {end_pred - start_pred:.2f} seconds")
print(f" Model Size:     {model_size:.2f} MB")
print("="*50)