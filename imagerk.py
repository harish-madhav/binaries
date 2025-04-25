import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


train_dir = 'images/train'
val_dir = 'images/val'

datagen = ImageDataGenerator(rescale=1./255)

train = datagen.flow_from_directory(train_dir, target_size=(64, 64), batch_size=32, class_mode='categorical')
val = datagen.flow_from_directory(val_dir, target_size=(64, 64), batch_size=32, class_mode='categorical')


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(train.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train, validation_data=val, epochs=5)


loss, accuracy = model.evaluate(val)
print("Validation Accuracy:", accuracy)