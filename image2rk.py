# pip install tensorflow scikit-learn matplotlib numpy

import zipfile
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np



extract_path = r'C:\Users\shant\OneDrive\Desktop\model ds\jelly_dataset'

print("Contents of dataset directory:", os.listdir(extract_path))


entries = os.listdir(extract_path)
if len(entries) == 1 and os.path.isdir(os.path.join(extract_path, entries[0])):
    extract_path = os.path.join(extract_path, entries[0])
    print(" Adjusted extract_path to:", extract_path)

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

img_size = (128, 128)
batch = 32

train_set = datagen.flow_from_directory(
    extract_path,
    target_size=img_size,
    batch_size=batch,
    class_mode='categorical',
    subset='training'
)

val_set = datagen.flow_from_directory(
    extract_path,
    target_size=img_size,
    batch_size=batch,
    class_mode='categorical',
    subset='validation'
)


model = models.Sequential([
    layers.Input(shape=(*img_size, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_set.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

if train_set.samples == 0 or val_set.samples == 0:
    print(" No training/validation samples found. Please check folder structure.")
    exit()

history = model.fit(train_set, epochs=10, validation_data=val_set)


val_loss, val_acc = model.evaluate(val_set)
print(f"\n Validation Accuracy: {val_acc * 100:.2f}%")

Y_pred = model.predict(val_set)
y_pred = np.argmax(Y_pred, axis=1)
true_classes = val_set.classes
class_labels = list(val_set.class_indices.keys())

print("\nClassification Report:")
print(classification_report(true_classes, y_pred, target_names=class_labels))

cm = confusion_matrix(true_classes, y_pred)
print("\nConfusion Matrix:")
print(cm)

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()