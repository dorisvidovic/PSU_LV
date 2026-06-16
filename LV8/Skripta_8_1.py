from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train_s = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test_s  = x_test.reshape(-1, 28, 28, 1) / 255.0

y_train_s = to_categorical(y_train, num_classes=10)
y_test_s  = to_categorical(y_test,  num_classes=10)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu',
                  input_shape=(28, 28, 1)),

    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),

    layers.Dense(64, activation='relu'),

    layers.Dropout(0.3),

    layers.Dense(10, activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

my_callbacks = [
    callbacks.TensorBoard(log_dir='logs', update_freq=100),
    callbacks.ModelCheckpoint(
        filepath='best_model.keras',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
]

history = model.fit(
    x_train_s, y_train_s,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    callbacks=my_callbacks,
    verbose=1
)

best_model = models.load_model('best_model.keras')

train_loss, train_acc = best_model.evaluate(x_train_s, y_train_s, verbose=0)
test_loss,  test_acc  = best_model.evaluate(x_test_s,  y_test_s,  verbose=0)

print(f"Točnost na skupu za učenje:   {train_acc:.4f}")
print(f"Točnost na skupu za testiranje: {test_acc:.4f}")

y_train_pred = np.argmax(best_model.predict(x_train_s, verbose=0), axis=1)
y_test_pred  = np.argmax(best_model.predict(x_test_s,  verbose=0), axis=1)

cm_train = confusion_matrix(y_train, y_train_pred)
cm_test  = confusion_matrix(y_test,  y_test_pred)

def plot_cm(cm, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel('Predviđena klasa')
    ax.set_ylabel('Stvarna klasa')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

    for i in range(10):
        for j in range(10):
            ax.text(
                j, i, cm[i, j],
                ha='center',
                va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black',
                fontsize=8
            )

    plt.colorbar(im)
    plt.tight_layout()
    plt.show()

plot_cm(cm_train, 'Matrica zabune — TRAIN')
plot_cm(cm_test,  'Matrica zabune — TEST')