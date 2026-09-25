import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import f1_score

print("Loading CIFAR-10 data...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Build a simple CNN from scratch
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\nStarting baseline training (5 epochs)...")
start_train = time.time()
model.fit(x_train, y_train, epochs=5, validation_split=0.1, batch_size=64)
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
model.save('baseline.keras')
model_size = os.path.getsize('baseline.keras') / (1024 * 1024) # MB

print("\n" + "="*50)
print(" BASELINE METRICS (From-Scratch CNN)")
print("="*50)
print(f" Accuracy:       {accuracy:.4f}")
print(f" F1 Score:       {f1:.4f}")
print(f" Train Time:     {end_train - start_train:.2f} seconds")
print(f" Predict Time:   {end_pred - start_pred:.2f} seconds")
print(f" Model Size:     {model_size:.2f} MB")
print("="*50)