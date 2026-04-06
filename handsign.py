# ============================================================
# HandSign — Sign Language Digit Recognizer
# Real-world problem: Help deaf/mute learners recognize and
# practice sign language digits (0–9) using deep learning.
#
# Dataset: Sign Language MNIST (Kaggle)
# https://www.kaggle.com/datasets/datamunge/sign-language-mnist
#
# Author : Sakshi Saraswat | GWECA, Ajmer
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Flatten,
                                     BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")

# ── 1. Configuration ─────────────────────────────────────────
IMG_SIZE    = 28          # 28x28 grayscale images
NUM_CLASSES = 10          # digits 0–9
BATCH_SIZE  = 64
EPOCHS      = 30
LR          = 0.001
SEED        = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Class labels: digits 0–9
CLASS_NAMES = [str(i) for i in range(10)]

# ── 2. Load Data ─────────────────────────────────────────────
# Download from Kaggle:
#   kaggle datasets download datamunge/sign-language-mnist
# Place sign_mnist_train.csv and sign_mnist_test.csv in same folder.

print("Loading Sign Language MNIST dataset...")
train_df = pd.read_csv("sign_mnist_train.csv")
test_df  = pd.read_csv("sign_mnist_test.csv")

print(f"Train samples : {len(train_df)}")
print(f"Test  samples : {len(test_df)}")
print(f"Classes       : {sorted(train_df['label'].unique())}")

# ── 3. Preprocess ─────────────────────────────────────────────
def preprocess(df):
    labels  = df["label"].values
    pixels  = df.drop("label", axis=1).values.astype("float32") / 255.0
    images  = pixels.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    return images, labels

X_train_full, y_train_full = preprocess(train_df)
X_test,       y_test       = preprocess(test_df)

# Train / Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.15, random_state=SEED, stratify=y_train_full
)

# One-hot encode labels
lb = LabelBinarizer()
y_train_enc = lb.fit_transform(y_train)
y_val_enc   = lb.transform(y_val)
y_test_enc  = lb.transform(y_test)

print(f"\nTrain : {X_train.shape}, Val : {X_val.shape}, Test : {X_test.shape}")

# ── 4. Visualize Sample Images ────────────────────────────────
def plot_samples(X, y, n=10):
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle("HandSign — Sample Sign Language Digits", fontsize=14, fontweight="bold")
    for i, ax in enumerate(axes.flat):
        ax.imshow(X[i].reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
        ax.set_title(f"Digit: {y[i]}", fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("sample_images.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: sample_images.png")

plot_samples(X_train, y_train)

# ── 5. Class Distribution ─────────────────────────────────────
def plot_distribution(y, title):
    unique, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(10, 4))
    bars = plt.bar(unique, counts, color=sns.color_palette("viridis", len(unique)))
    plt.title(f"{title}", fontsize=13, fontweight="bold")
    plt.xlabel("Digit Class")
    plt.ylabel("Sample Count")
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 str(count), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: class_distribution.png")

plot_distribution(y_train, "HandSign — Training Set Class Distribution")

# ── 6. Model Architecture ─────────────────────────────────────
# Fully connected neural network (no CNN) as per the project brief.
# Good baseline for 28x28 grayscale sign images.

def build_model(input_shape, num_classes, dropout_rate=0.4):
    model = Sequential([
        # Flatten the 28x28x1 image to 784 features
        Flatten(input_shape=input_shape, name="flatten"),

        # Hidden layer 1
        Dense(512, activation="relu", name="dense_1"),
        BatchNormalization(name="bn_1"),
        Dropout(dropout_rate, name="dropout_1"),

        # Hidden layer 2
        Dense(256, activation="relu", name="dense_2"),
        BatchNormalization(name="bn_2"),
        Dropout(dropout_rate, name="dropout_2"),

        # Hidden layer 3
        Dense(128, activation="relu", name="dense_3"),
        BatchNormalization(name="bn_3"),
        Dropout(dropout_rate * 0.75, name="dropout_3"),

        # Hidden layer 4
        Dense(64, activation="relu", name="dense_4"),
        Dropout(0.2, name="dropout_4"),

        # Output layer
        Dense(num_classes, activation="softmax", name="output"),
    ], name="HandSign_NN")

    return model

model = build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES)
model.summary()

# ── 7. Compile ────────────────────────────────────────────────
model.compile(
    optimizer=Adam(learning_rate=LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ── 8. Callbacks ──────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=7,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=3, min_lr=1e-6, verbose=1),
]

# ── 9. Train ──────────────────────────────────────────────────
print("\nTraining HandSign model...")
history = model.fit(
    X_train, y_train_enc,
    validation_data=(X_val, y_val_enc),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ── 10. Evaluate ──────────────────────────────────────────────
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test_enc, verbose=0)
print(f"\nTest Accuracy : {test_acc * 100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")

# ── 11. Training Curves ───────────────────────────────────────
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("HandSign — Training History", fontsize=14, fontweight="bold")

    ax1.plot(history.history["accuracy"],     label="Train Accuracy", color="#2196F3")
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy",   color="#4CAF50")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(history.history["loss"],     label="Train Loss", color="#F44336")
    ax2.plot(history.history["val_loss"], label="Val Loss",   color="#FF9800")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: training_curves.png")

plot_history(history)

# ── 12. Confusion Matrix ──────────────────────────────────────
def plot_confusion_matrix(X, y_true, model, lb):
    y_pred_probs = model.predict(X, verbose=0)
    y_pred       = np.argmax(y_pred_probs, axis=1)
    y_true_dec   = lb.inverse_transform(
        np.eye(NUM_CLASSES)[y_true.astype(int)] if y_true.ndim == 1
        else y_true
    )

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("HandSign — Confusion Matrix (Test Set)", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Digit"); plt.ylabel("True Digit")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: confusion_matrix.png")

    print("\nClassification Report:")
    print(classification_report(y_test, np.argmax(
        model.predict(X_test, verbose=0), axis=1),
        target_names=[f"Digit {i}" for i in range(NUM_CLASSES)]
    ))

plot_confusion_matrix(X_test, y_test, model, lb)

# ── 13. Predict & Visualize Single Samples ────────────────────
def predict_samples(X, y_true, model, n=10):
    indices = np.random.choice(len(X), n, replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle("HandSign — Predictions vs Ground Truth", fontsize=13, fontweight="bold")

    for i, (ax, idx) in enumerate(zip(axes.flat, indices)):
        img = X[idx]
        true_label = y_true[idx]
        pred_probs = model.predict(img[np.newaxis], verbose=0)[0]
        pred_label = np.argmax(pred_probs)
        confidence = pred_probs[pred_label] * 100

        ax.imshow(img.reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
        color = "green" if pred_label == true_label else "red"
        ax.set_title(f"True: {true_label} | Pred: {pred_label}\n{confidence:.1f}%",
                     fontsize=9, color=color)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("predictions.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: predictions.png")

predict_samples(X_test, y_test, model)

# ── 14. Save Model ────────────────────────────────────────────
model.save("handsign_model.h5")
print("\nModel saved as handsign_model.h5")

print("""
╔══════════════════════════════════════════════════╗
║           HandSign — Results Summary             ║
╠══════════════════════════════════════════════════╣
║  Problem   : Sign language digit recognition     ║
║  Impact    : Assistive tech for deaf learners    ║
║  Dataset   : Sign Language MNIST (Kaggle)        ║
║  Model     : Fully connected Neural Network      ║
║  Output    : Predicted digit class (0–9)         ║
╚══════════════════════════════════════════════════╝
""")
