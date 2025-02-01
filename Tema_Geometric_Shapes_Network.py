import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from PIL import Image, ImageDraw
from tensorflow import keras as keras
from keras import layers
from keras.models import Sequential
import pathlib
import math
from collections import Counter
import random
import pandas as pd
import np_utils
import sklearn as sk
from sklearn.cluster import KMeans
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76, rgb2hsv


#cream pozele cu formele geometrice
w,h=96,96
a=12

#cream foldere pentru fiecare forma geometrica
directory = "rectangle"
parentDir = "D:/work/IntelACLabs/geometric_shapes/"
path = os.path.join(parentDir,directory)
os.mkdir(path)
directory = "triangle"
parentDir = "D:/work/IntelACLabs/geometric_shapes/"
path = os.path.join(parentDir,directory)
os.mkdir(path)
directory = "ellipse"
parentDir = "D:/work/IntelACLabs/geometric_shapes/"
path = os.path.join(parentDir,directory)
os.mkdir(path)

colors = ["red","green","blue"]

for i in range(1000):
    image = Image.new('RGB', (w,h), 'white') 
    draw = ImageDraw.Draw(image)
    x1 = np.random.randint(a,w-a)
    x2 = np.random.randint(a,w-a)
    y1 = np.random.randint(a,h-a)
    y2 = np.random.randint(a,h-a)
    draw.rectangle([(x1, y1), (x2 ,y2)], fill=random.choice(colors)) 
    image.save('D:/work/IntelACLabs/geometric_shapes/rectangle/rectangle_'+str(i)+'.png')

for i in range(1000):
    image = Image.new('RGB', (w,h), 'white')
    draw = ImageDraw.Draw(image)
    x1 = np.random.randint(a,w-a)
    x2 = np.random.randint(a,w-a)
    x3 = np.random.randint(a,w-a)
    y1 = np.random.randint(a,h-a)
    y2 = np.random.randint(a,h-a)
    y3 = np.random.randint(a,h-a)
    draw.polygon([(x1,y1), (x2,y2), (x3,y3)], fill=random.choice(colors))
    image.save('D:/work/IntelACLabs/geometric_shapes/triangle/triangle_'+str(i)+'.png')

for i in range(1000):
    image = Image.new('RGB', (w,h), 'white') 
    draw = ImageDraw.Draw(image)
    x1 = np.random.randint(a,w-a)
    x2 = np.random.randint(a,w-a)
    y1 = np.random.randint(a,h-a)
    y2 = np.random.randint(a,h-a)
    eX, eY = 30, 60
    g=x1/2 - eX/2
    L=y1/2 - eY/2
    j=x2/2 + eX/2
    k=y2/2 + eY/2
    if g not in range(a, w-a):
        if g<a: g=g+a
        if g>w-a: g=g-a
    if L not in range(a, w-a):
        if L<a: L=L+a
        if L>w-a: L=L-a
    if j not in range(a, w-a):
        if j<a: j=j+a
        if j>w-a: j=j-a
    if k not in range(a, w-a):
        if k<a: k=k+a
        if k>w-a: k=k-a
    bbox =  (g,L,j,k)
    draw.ellipse(bbox, fill=random.choice(colors)) 
    image.save('D:/work/IntelACLabs/geometric_shapes/ellipse/ellipse_'+str(i)+'.png')

#importam folderul cu pozele dupa care vom antrena programul
data_dir = pathlib.Path("geometric_shapes")

geom_shape=[0,1,2] #0-ellipse 1-rectangle 2-triangle
if geom_shape==0:
    classname="ellipse"
elif geom_shape==1:
  classname="rectangle"
elif geom_shape==2:
  classname="triangle"

#cream un set de date
batch_size = 32
img_height = 96
img_width = 96
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
  )

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#salvam denumirile formelor geometrice
class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

#cream modelul secvential
num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, (3,3), padding='same', activation='relu'),
  layers.MaxPooling2D((2,2)),
  layers.Conv2D(32, (3,3), padding='same', activation='relu'),
  layers.MaxPooling2D((2,2)),
  layers.Conv2D(64, (3,3), padding='same', activation='relu'),
  layers.MaxPooling2D((2,2)),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

#il compilam
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#verificam modelul
#model.summary()

#antrenam modelul
epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

#data augmentation
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

#repetam
model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, (3,3), padding='same', activation='relu'),
  layers.MaxPooling2D((2,2)),
  layers.Conv2D(32, (3,3), padding='same', activation='relu'),
  layers.MaxPooling2D((2,2)),
  layers.Conv2D(64, (3,3), padding='same', activation='relu'),
  layers.MaxPooling2D((2,2)),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs = 10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


#introducem pozele de input si le setam marimea ca fiind 96x96
img1 = cv2.imread("D:/work/IntelACLabs/input/1.jpg")
cv2.resize(img1,(96,96))
cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread("D:/work/IntelACLabs/input/2.jpg")
cv2.resize(img2,(96,96))
cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img3 = cv2.imread("D:/work/IntelACLabs/input/3.jpg")
cv2.resize(img3,(96,96))
cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
img4 = cv2.imread("D:/work/IntelACLabs/input/4.jpg")
cv2.resize(img4,(96,96))
cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
img5 = cv2.imread("D:/work/IntelACLabs/input/5.jpg")
cv2.resize(img5,(96,96))
cv2.cvtColor(img5, cv2.COLOR_BGR2RGB)
img6 = cv2.imread("D:/work/IntelACLabs/input/6.jpg")
cv2.resize(img6,(96,96))
cv2.cvtColor(img6, cv2.COLOR_BGR2RGB)

#introducem imaginile de input si le setam marimea ca fiind 96x96
#afisam imaginile, forma lor, culoarea, procentul ocupat din imagine

#culoarea
b = img1[:, :, :1]
g = img1[:, :, 1:2]
r = img1[:, :, 2:]
b_mean = np.mean(b)
g_mean = np.mean(g)
r_mean = np.mean(r)
#aria
im=Image.open("D:/work/IntelACLabs/input/1.jpg")
bg = im.getpixel((0,0))
bg_count = next(n for n,c in im.getcolors(w*h) if c==bg)
img_count = w*h - bg_count
img_percent = img_count*100.0/w/h
#forma
imgp="D:/work/IntelACLabs/input/1.jpg"
imgp = tf.keras.utils.load_img(
  imgp, target_size=(w, h)
)
img_array = tf.keras.utils.img_to_array(imgp)
img_array = tf.expand_dims(img_array, 0) # Create a batch
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
imgplot = plt.imshow(imgp)
plt.text(w-20, h-20, class_names[np.argmax(score)], bbox=dict(fill=False, edgecolor='red', linewidth=2))
if (b_mean > g_mean and b_mean > r_mean):
  plt.text(w-40, h-40, "blue", bbox=dict(fill=False, edgecolor='red', linewidth=2))
if (g_mean > r_mean and g_mean > b_mean):
  plt.text(w-40, h-40, "green", bbox=dict(fill=False, edgecolor='red', linewidth=2))
else:
  plt.text(w-40, h-40, "red", bbox=dict(fill=False, edgecolor='red', linewidth=2))
plt.text(w-60, h-60, str(img_percent), bbox=dict(fill=False, edgecolor='red', linewidth=2))
plt.show()

#culoarea
b = img2[:, :, :1]
g = img2[:, :, 1:2]
r = img2[:, :, 2:]
b_mean = np.mean(b)
g_mean = np.mean(g)
r_mean = np.mean(r)
#aria
im=Image.open("D:/work/IntelACLabs/input/2.jpg")
bg = im.getpixel((0,0))
bg_count = next(n for n,c in im.getcolors(w*h) if c==bg)
img_count = w*h - bg_count
img_percent = img_count*100.0/w/h
#forma
imgp="D:/work/IntelACLabs/input/2.jpg"
imgp = tf.keras.utils.load_img(
  imgp, target_size=(w, h)
)
img_array = tf.keras.utils.img_to_array(imgp)
img_array = tf.expand_dims(img_array, 0) # Create a batch
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
imgplot = plt.imshow(imgp)
plt.text(w-20, h-20, class_names[np.argmax(score)], bbox=dict(fill=False, edgecolor='red', linewidth=2))
if (b_mean > g_mean and b_mean > r_mean):
  plt.text(w-40, h-40, "blue", bbox=dict(fill=False, edgecolor='red', linewidth=2))
if (g_mean > r_mean and g_mean > b_mean):
  plt.text(w-40, h-40, "green", bbox=dict(fill=False, edgecolor='red', linewidth=2))
else:
  plt.text(w-40, h-40, "red", bbox=dict(fill=False, edgecolor='red', linewidth=2))
plt.text(w-60, h-60, str(img_percent), bbox=dict(fill=False, edgecolor='red', linewidth=2))
plt.show()

#culoarea
b = img3[:, :, :1]
g = img3[:, :, 1:2]
r = img3[:, :, 2:]
b_mean = np.mean(b)
g_mean = np.mean(g)
r_mean = np.mean(r)
#aria
im=Image.open("D:/work/IntelACLabs/input/3.jpg")
bg = im.getpixel((0,0))
bg_count = next(n for n,c in im.getcolors(w*h) if c==bg)
img_count = w*h - bg_count
img_percent = img_count*100.0/w/h
#forma
imgp="D:/work/IntelACLabs/input/3.jpg"
imgp = tf.keras.utils.load_img(
  imgp, target_size=(w, h)
)
img_array = tf.keras.utils.img_to_array(imgp)
img_array = tf.expand_dims(img_array, 0) # Create a batch
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
imgplot = plt.imshow(imgp)
plt.text(w-20, h-20, class_names[np.argmax(score)], bbox=dict(fill=False, edgecolor='red', linewidth=2))
if (b_mean > g_mean and b_mean > r_mean):
  plt.text(w-40, h-40, "blue", bbox=dict(fill=False, edgecolor='red', linewidth=2))
if (g_mean > r_mean and g_mean > b_mean):
  plt.text(w-40, h-40, "green", bbox=dict(fill=False, edgecolor='red', linewidth=2))
else:
  plt.text(w-40, h-40, "red", bbox=dict(fill=False, edgecolor='red', linewidth=2))
plt.text(w-60, h-60, str(img_percent), bbox=dict(fill=False, edgecolor='red', linewidth=2))
plt.show()

#culoarea
b = img4[:, :, :1]
g = img4[:, :, 1:2]
r = img4[:, :, 2:]
b_mean = np.mean(b)
g_mean = np.mean(g)
r_mean = np.mean(r)
#aria
im=Image.open("D:/work/IntelACLabs/input/4.jpg")
bg = im.getpixel((0,0))
bg_count = next(n for n,c in im.getcolors(w*h) if c==bg)
img_count = w*h - bg_count
img_percent = img_count*100.0/w/h
#forma
imgp="D:/work/IntelACLabs/input/4.jpg"
imgp = tf.keras.utils.load_img(
  imgp, target_size=(w, h)
)
img_array = tf.keras.utils.img_to_array(imgp)
img_array = tf.expand_dims(img_array, 0) # Create a batch
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
imgplot = plt.imshow(imgp)
plt.text(w-20, h-20, class_names[np.argmax(score)], bbox=dict(fill=False, edgecolor='red', linewidth=2))
if (b_mean > g_mean and b_mean > r_mean):
  plt.text(w-40, h-40, "blue", bbox=dict(fill=False, edgecolor='red', linewidth=2))
if (g_mean > r_mean and g_mean > b_mean):
  plt.text(w-40, h-40, "green", bbox=dict(fill=False, edgecolor='red', linewidth=2))
else:
  plt.text(w-40, h-40, "red", bbox=dict(fill=False, edgecolor='red', linewidth=2))
plt.text(w-60, h-60, str(img_percent), bbox=dict(fill=False, edgecolor='red', linewidth=2))
plt.show()

#culoarea
b = img5[:, :, :1]
g = img5[:, :, 1:2]
r = img5[:, :, 2:]
b_mean = np.mean(b)
g_mean = np.mean(g)
r_mean = np.mean(r)
#aria
im=Image.open("D:/work/IntelACLabs/input/5.jpg")
bg = im.getpixel((0,0))
bg_count = next(n for n,c in im.getcolors(w*h) if c==bg)
img_count = w*h - bg_count
img_percent = img_count*100.0/w/h
#forma
imgp="D:/work/IntelACLabs/input/5.jpg"
imgp = tf.keras.utils.load_img(
  imgp, target_size=(w, h)
)
img_array = tf.keras.utils.img_to_array(imgp)
img_array = tf.expand_dims(img_array, 0) # Create a batch
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
imgplot = plt.imshow(imgp)
plt.text(w-20, h-20, class_names[np.argmax(score)], bbox=dict(fill=False, edgecolor='red', linewidth=2))
if (b_mean > g_mean and b_mean > r_mean):
  plt.text(w-40, h-40, "blue", bbox=dict(fill=False, edgecolor='red', linewidth=2))
if (g_mean > r_mean and g_mean > b_mean):
  plt.text(w-40, h-40, "green", bbox=dict(fill=False, edgecolor='red', linewidth=2))
else:
  plt.text(w-40, h-40, "red", bbox=dict(fill=False, edgecolor='red', linewidth=2))
plt.text(w-60, h-60, str(img_percent), bbox=dict(fill=False, edgecolor='red', linewidth=2))
plt.show()

#culoarea
b = img6[:, :, :1]
g = img6[:, :, 1:2]
r = img6[:, :, 2:]
b_mean = np.mean(b)
g_mean = np.mean(g)
r_mean = np.mean(r)
#aria
im=Image.open("D:/work/IntelACLabs/input/6.jpg")
bg = im.getpixel((0,0))
bg_count = next(n for n,c in im.getcolors(w*h) if c==bg)
img_count = w*h - bg_count
img_percent = img_count*100.0/w/h
#forma
imgp="D:/work/IntelACLabs/input/6.jpg"
imgp = tf.keras.utils.load_img(
  imgp, target_size=(w, h)
)
img_array = tf.keras.utils.img_to_array(imgp)
img_array = tf.expand_dims(img_array, 0) # Create a batch
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
imgplot = plt.imshow(imgp)
plt.text(w-20, h-20, class_names[np.argmax(score)], bbox=dict(fill=False, edgecolor='red', linewidth=2))
if (b_mean > g_mean and b_mean > r_mean):
  plt.text(w-40, h-40, "blue", bbox=dict(fill=False, edgecolor='red', linewidth=2))
if (g_mean > r_mean and g_mean > b_mean):
  plt.text(w-40, h-40, "green", bbox=dict(fill=False, edgecolor='red', linewidth=2))
else:
  plt.text(w-40, h-40, "red", bbox=dict(fill=False, edgecolor='red', linewidth=2))
plt.text(w-60, h-60, str(img_percent), bbox=dict(fill=False, edgecolor='red', linewidth=2))
plt.show()