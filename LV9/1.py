import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


TRAIN_DIR  = "gtsrb/Train"
TEST_DIR   = "gtsrb/Test"
IMG_SIZE   = (48, 48)
BATCH_SIZE = 32
NUM_CLASSES = 43
EPOCHS     = 20
SEED       = 123


train_ds = image_dataset_from_directory(
    directory=TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    subset="training",
    seed=SEED,
    validation_split=0.2,
)

validation_ds = image_dataset_from_directory(
    directory=TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    subset="validation",
    seed=SEED,
    validation_split=0.2,
)

test_ds = image_dataset_from_directory(
    directory=TEST_DIR,
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False,
)

class_names = train_ds.class_names
print(f"Klase (", len(class_names), "): ", class_names)

# Prefetch & cache za bržu obradu
AUTOTUNE = tf.data.AUTOTUNE
train_ds      = train_ds.cache().prefetch(AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(AUTOTUNE)
test_ds       = test_ds.cache().prefetch(AUTOTUNE)

def build_model(input_shape=(48, 48, 3), num_classes=NUM_CLASSES):
    model = models.Sequential(name="GTSRB_CNN")

    # Normalizacija piksela [0, 255] -> [0, 1]
    model.add(layers.Rescaling(1.0/255, input_shape=input_shape))

    # Tri konvolucijska bloka s brojem filtera 32, 64, 128
    for filters in [32, 64, 128]:
        model.add(layers.Conv2D(filters, (3, 3), strides=1,
                                padding="same",  activation="relu"))
        model.add(layers.Conv2D(filters, (3, 3), strides=1,
                                padding="valid", activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(layers.Dropout(0.2))

    # Klasifikacijski dio
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model

model = build_model()
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()  # očekivano: Total params = 1,358,155

os.makedirs("models", exist_ok=True)
os.makedirs("logs",   exist_ok=True)

log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

cb_checkpoint = callbacks.ModelCheckpoint(
    filepath="models/best_gtsrb.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1,
)
cb_tensorboard = callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
)
cb_early = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)


history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS,
    callbacks=[cb_checkpoint, cb_tensorboard, cb_early],
    verbose=1,
)


print("\nUčitavam najbolji model...")
best_model = tf.keras.models.load_model("models/best_gtsrb.keras")

test_loss, test_acc = best_model.evaluate(test_ds, verbose=1)
print(f"\nTočnost na testnom skupu: {test_acc:.4f}")
print(f"Gubitak na testnom skupu: {test_loss:.4f}")

# Predikcije i stvarne oznake
y_true, y_pred = [], []
for images, labels in test_ds:
    preds = best_model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds,  axis=1))
    y_true.extend(np.argmax(labels, axis=1))
y_true, y_pred = np.array(y_true), np.array(y_pred)


cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=False, fmt="d", cmap="Oranges",
            xticklabels=class_names, yticklabels=class_names)
plt.title(f"Matrica zabune — test acc = {test_acc:.4f}", fontsize=14)
plt.xlabel("Predviđena klasa")
plt.ylabel("Stvarna klasa")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

print("\n=== Classification report ===")
print(classification_report(y_true, y_pred, target_names=class_names))


fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history["accuracy"],     label="train")
ax[0].plot(history.history["val_accuracy"], label="val")
ax[0].set_title("Točnost"); ax[0].legend(); ax[0].grid(True, alpha=0.3)
ax[1].plot(history.history["loss"],     label="train")
ax[1].plot(history.history["val_loss"], label="val")
ax[1].set_title("Gubitak"); ax[1].legend(); ax[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()