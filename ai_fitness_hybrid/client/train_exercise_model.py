# client/train_exercise_model.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Load dataset
X = np.load("exercise_X.npy")
y = np.load("exercise_y.npy")
labels = np.load("exercise_labels.npy", allow_pickle=True).item()

num_classes = len(labels)

print("X:", X.shape)
print("y:", y.shape)
print("Labels:", labels)

# Proper stratified split
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train:", X_train.shape)
print("Val:", X_val.shape)

# Build model
model = Sequential([
    GRU(64, return_sequences=True, input_shape=(30, 24)),
    Dropout(0.3),
    GRU(32),
    Dense(32, activation="relu"),
    Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

# Train
model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop]
)

# Save model
import os
os.makedirs("../models", exist_ok=True)
model.save("../models/exercise_classifier.keras")

print("✅ Exercise classification model saved")
