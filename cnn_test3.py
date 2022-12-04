
from curses import flash
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
from keras.metrics import categorical_crossentropy
from keras.optimizers import Adam
import tensorflow as tf
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import matplotlib_inline
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn import *


#Script for organizing the data set
os.chdir('/Users/corinnewatson/Desktop/dogs-vs-cats')
if os.path.isdir('train/dog') is False:
    os.makedirs('train/dog')
    os.makedirs('train/cat')
    os.makedirs('valid/dog')
    os.makedirs('valid/cat')
    os.makedirs('test/dog')
    os.makedirs('test/cat')

    for c in random.sample(glob.glob('cat*'), 500):
        shutil.move(c, 'train/cat')
    for c in random.sample(glob.glob('dog*'), 500):
        shutil.move(c, 'train/dog')
    for c in random.sample(glob.glob('cat*'), 100):
        shutil.move(c, 'valid/cat')
    for c in random.sample(glob.glob('dog*'), 100):
        shutil.move(c, 'valid/dog')
    for c in random.sample(glob.glob('cat*'), 50):
        shutil.move(c, 'test/cat')
    for c in random.sample(glob.glob('dog*'), 50):
        shutil.move(c, 'test/dog')


train_path = '/Users/corinnewatson/Desktop/dogs-vs-cats/train'
test_path = '/Users/corinnewatson/Desktop/dogs-vs-cats/test'
valid_path = '/Users/corinnewatson/Desktop/dogs-vs-cats/valid'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10, shuffle=False)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=valid_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10)

imgs, labels = next(train_batches)

def plotImages(images_arr):
    fig, axes = plt.subplots(1,10,figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# plotImages(imgs)
# print(labels)

# Set the cnn instance 
model = Sequential([Convolution2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(224,224,3)),
MaxPooling2D(pool_size=(2,2), strides=2), Convolution2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'),
MaxPooling2D(pool_size=(2,2), strides=2), Flatten(), Dense(units=2, activation='softmax')])


model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_batches, validation_data=valid_batches, epochs=5, verbose=2)

test_imgs, test_labels = next(test_batches)
# plotImages(test_imgs)
# print(test_labels)

test_batches.classes

predictions = model.predict(x=test_batches, verbose=0)
np.round(predictions)


cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=1))

def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion Matrixxx', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation= 'nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion Matric w/o normalization")

    print(cm)

    thresh = cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment = 'center', color="white" if cm[i,j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('true label')
    plt.xlabel("Predicted label")
    print("HIT CONF MATRIX CODE")

test_batches.class_indices


cm_plot_labels = ['cat', 'dog']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title= 'Confusion Matrixxx')

