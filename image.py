import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense

gen=ImageDataGenerator(rescale=1./255,validation_split=0.2)
td=gen.flow_from_directory('Animals',target_size=(64,64),batch_size=32,class_mode='categorical',subset='training')
vd=gen.flow_from_directory('Animals',target_size=(64,64),batch_size=32,class_mode='categorical',subset='validation')

m=Sequential([
        Conv2D(32, (3, 3),activation='relu',input_shape=(64,64,3)),
        MaxPooling2D(2,2),
        Conv2D(64,(3,3),activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128,activation='relu'),
        Dense(td.num_classes,activation='softmax')])
m.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
m.fit(td,validation_data=vd,epochs=5)
l,a=m.evaluate(vd)
print(f"Validation data {a:.2f}%")

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

class_names =['cats','dogs','snakes']

def pre(ip):
  im=image.load_img(ip,target_size=(64, 64))
  ita=image.img_to_array(im)
  ita=ita/255.00
  ita=np.expand_dims(ita,axis=0)
  
  pr=m.predict(ita)
  prc=class_names[np.argmax(pr)]
  con=np.max(pr)

  print(f"prediction class : {prc}")
  print(f"confidence : {con}")
pre('image.jpg')
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense

gen=ImageDataGenerator(rescale=1./255,validation_split=0.2)
td=gen.flow_from_directory('Animals',target_size=(64,64),batch_size=32,class_mode='categorical',subset='training')
vd=gen.flow_from_directory('Animals',target_size=(64,64),batch_size=32,class_mode='categorical',subset='validation')

m=Sequential([
        Conv2D(32, (3, 3),activation='relu',input_shape=(64,64,3)),
        MaxPooling2D(2,2),
        Conv2D(64,(3,3),activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128,activation='relu'),
        Dense(td.num_classes,activation='softmax')])
m.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
m.fit(td,validation_data=vd,epochs=5)
l,a=m.evaluate(vd)
print(f"Validation data {a:.2f}%")

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

class_names =['cats','dogs','snakes']

def pre(ip):
  im=image.load_img(ip,target_size=(64, 64))
  ita=image.img_to_array(im)
  ita=ita/255.00
  ita=np.expand_dims(ita,axis=0)
  
  pr=m.predict(ita)
  prc=class_names[np.argmax(pr)]
  con=np.max(pr)

  print(f"prediction class : {prc}")
  print(f"confidence : {con}")
pre('image.jpg')