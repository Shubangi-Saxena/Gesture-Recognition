import os
from os import listdir
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import keras
from keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator#gives a label automatically
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import joblib

train=ImageDataGenerator(rescale=1/255)
validation=ImageDataGenerator(rescale=1/255)


path=r"G:\.shortcut-targets-by-id\1YcF8QImbD1Lq3FezjLNu7lh1tmeVIF1k\Gesture Recognition\ISL\output\train"
train_classes=list()
for i in os.listdir(path):
    if(len(i)==1):
        train_classes.append(i)
train_dataset=train.flow_from_directory(
    path,
    target_size = (128, 128),
    batch_size=128,
    #subset='training',
    classes=train_classes,
    class_mode='categorical',
    shuffle=True
)

path=r"G:\.shortcut-targets-by-id\1YcF8QImbD1Lq3FezjLNu7lh1tmeVIF1k\Gesture Recognition\ISL\output\val"
valid_classes=list()
for i in os.listdir(path):
    if(len(i)==1):
        valid_classes.append(i)
valid_dataset=train.flow_from_directory(
    path,
    target_size = (128, 128),
    batch_size=128,
    #subset='validating',
    classes=valid_classes,
    class_mode='categorical',
    shuffle=True
)

path=r"G:\.shortcut-targets-by-id\1YcF8QImbD1Lq3FezjLNu7lh1tmeVIF1k\Gesture Recognition\ISL\output\test"
test_classes=list()
for i in os.listdir(path):
    if(len(i)==1):
        test_classes.append(i)
test_dataset=train.flow_from_directory(
    path,
    target_size = (128, 128),
    batch_size=128,
    classes=test_classes,
    class_mode='categorical',
    shuffle=True,
)

print("Model: ")
model=tf.keras.models.Sequential(
    [tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(128,128,3)),
     tf.keras.layers.MaxPool2D(),
     #
     tf.keras.layers.Conv2D(32,(3,3),kernel_regularizer=regularizers.l2(0.001),activation='relu'),
     tf.keras.layers.MaxPool2D(),
     #
     tf.keras.layers.Conv2D(16,(3,3),kernel_regularizer=regularizers.l2(0.001),activation='relu'),
     tf.keras.layers.MaxPool2D(),
     #
     tf.keras.layers.Conv2D(64,(3,3),kernel_regularizer=regularizers.l2(0.001),activation='relu'),
     tf.keras.layers.MaxPool2D(),
     #
     tf.keras.layers.Conv2D(64,(3,3),kernel_regularizer=regularizers.l2(0.001),activation='relu'),
     tf.keras.layers.MaxPool2D(),
     #
     tf.keras.layers.Flatten(),
     ##
     tf.keras.layers.Dense(512,activation='relu'),
     ##
     tf.keras.layers.Dense(35,activation='softmax')
     #
     ]
)

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

checkpoint_path = "G:\.shortcut-targets-by-id\1YcF8QImbD1Lq3FezjLNu7lh1tmeVIF1k\Gesture Recognition\ISL\output\checkpoint1.ckpt"
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
print("Fitting:")
history=model.fit(train_dataset,
          batch_size=128,
          #steps_per_epoch=10,
          epochs=1,
          callbacks=[cp_callback],
          validation_data=valid_dataset,
          verbose=1)

print("Done fitting")

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

#joblib.dump(model, 'cnn.npy')
model.save("my_model")

#neww= keras.models.load_model("my_model")
actual=list()
actual=['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

count=0
for i in range(99): #batch number
    for j in range(128):  #batch size
        img=test_dataset[i][0][j] #0-image
        img=img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
        pred=model.predict(img)
        #print(pred)
        index = np.argmax(pred[0]) #[0.1*10^-11,0.2*10^-3,0,1,0]
        #print(train_classes[index])
        actual_index=np.argmax(test_dataset[i][1][j]) #label
        if(actual[actual_index]==train_classes[index]):
            count+=1 #predictions are right

print(count/12672) #right pred/total

