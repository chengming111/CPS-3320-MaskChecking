

## 2. Import the Keras library and divide the dataset

import os

base_dir = '/Users/chengming/Desktop/datasets/mldata'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'va')
test_dir = os.path.join(base_dir, 'test')

train_mask_dir = os.path.join(train_dir, 'trainMask')
validation_mask_dir = os.path.join(validation_dir, 'vaMask')
test_mask_dir = os.path.join(test_dir, 'testMask')

train_unmask_dir = os.path.join(train_dir, 'trainUnMask')
validation_unmask_dir = os.path.join(validation_dir, 'vaUnMask')
test_unmask_dir = os.path.join(test_dir, 'testUnMask')


# Displays the number of images in each folder
print('total training mask images:', len(os.listdir(train_mask_dir)))
print('total training unmask images:', len(os.listdir(train_unmask_dir)))
print('total validation maskt images:', len(os.listdir(validation_mask_dir)))
print('total validation unmask images:', len(os.listdir(validation_unmask_dir)))
print('total test mask images:', len(os.listdir(test_mask_dir)))
print('total test unmask images:', len(os.listdir(test_unmask_dir)))


## 3. Build the network

from keras import layers
from keras import models
# Build a model
model = models.Sequential()
# Add layers
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Displays the output for each layer
model.summary()


## 4. Data preprocessing

from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


from keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=10,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=10,
        class_mode='binary')


for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


## Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=200,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)

model.save('mask_and_unmask_small_1.h5')


import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))


plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()



## 5. Using data enhancement

datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

import tensorflow as tf
# This is module with image preprocessing utilities
from keras.preprocessing import image

fnames = [os.path.join(train_mask_dir, fname) for fname in os.listdir(train_mask_dir)]
# We pick one image to "augment"
img_path = fnames[3]

# Read the image and resize it
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(150, 150))

# Convert it to a Numpy array with shape (150, 150, 3)
# x = image.img_to_array(img)
x = tf.keras.preprocessing.image.img_to_array(img)

# Reshape it to (1, 150, 150, 3)
x = x.reshape((1,) + x.shape)

# The .flow() command below generates batches of randomly transformed images.
# It will loop indefinitely, so we need to `break` the loop at some point!
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(tf.keras.preprocessing.image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break

plt.show()


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=10,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=10,
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)


model.save('mask_and_unmask_small_2.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


## 6. Use cameras to read faces for recognition

import cv2
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import dlib
from PIL import Image
model = load_model('/Users/chengming/MaskDetection/mask_and_unmask_small_2.h5')
detector = dlib.get_frontal_face_detector()
video=cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
def rec(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dets=detector(gray,1)
    if dets is not None:
        for face in dets:
            left=face.left()
            top=face.top()
            right=face.right()
            bottom=face.bottom()
            cv2.rectangle(img,(left,top),(right,bottom),(0,255,0),2)
            img1=cv2.resize(img[top:bottom,left:right],dsize=(150,150))
            img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
            img1 = np.array(img1)/255.
            img_tensor = img1.reshape(-1,150,150,3)
            prediction =model.predict(img_tensor)    
            if prediction[0][0]>0.5:
                result='unmask'
            else:
                result='mask'
            cv2.putText(img, result, (left,top), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Video', img)
while video.isOpened():
    res, img_rd = video.read()
    if not res:
        break
    rec(img_rd)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()

