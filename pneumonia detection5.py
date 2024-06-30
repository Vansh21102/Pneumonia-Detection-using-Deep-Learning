import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split

bacteria_path = "C:\\Users\\vansh\\OneDrive\\Desktop\\programs\\Pneumonia diagnosis\\train\\bacterial"
virus_path = "C:\\Users\\vansh\\OneDrive\\Desktop\\programs\\Pneumonia diagnosis\\train\\viral"
normal_path = "C:\\Users\\vansh\\OneDrive\\Desktop\\programs\\Pneumonia diagnosis\\train\\NORMAL"

evaluation_path = "C:\\Users\\vansh\\OneDrive\\Desktop\\programs\\Pneumonia diagnosis\\Test"

def create_testing_data():
    testing_data = []
    for img in os.listdir(evaluation_path):
        img_array = cv2.imread(os.path.join(evaluation_path, img))
        new_img = cv2.resize(img_array,(224,224))
        try:
            if "bacteria" in img:
                testing_data.append([new_img,2])
            elif "virus" in img:
                testing_data.append([new_img,1])
            elif "NORMAL" in img:
                testing_data.append([new_img,0])
        except:
            pass
    return testing_data

def create_training_data():
    training_data = []

    for img in os.listdir(bacteria_path):
        try:
            img_array = cv2.imread(os.path.join(bacteria_path, img))
            new_img = cv2.resize(img_array,(224,224))                       # cv2.resize() returns an image of the specified dimensions hence ALWAYS assign it to a variable
            training_data.append([new_img,2])                              # 2 for bacterial images
        except:
            pass

    for img in os.listdir(virus_path):
        try:
            viral_array = cv2.imread(os.path.join(virus_path, img))
            viral_img = cv2.resize(viral_array,(224,224))
            training_data.append([viral_img,1])                             # 1 for viral images       
        except:
            pass
    
    for img in os.listdir(normal_path):
        try:
            norm_array = cv2.imread(os.path.join(normal_path, img))
            norm_img = cv2.resize(norm_array,(224,224))
            training_data.append([norm_img,0])                             # 0 for normal images
        except:
            pass
    return training_data

train_data = create_training_data()
test_data = create_testing_data()

train_data, eval_data = train_test_split(train_data, test_size = 0.2, random_state = 42)

print(len(eval_data))
print(len(train_data))

# plt.imshow(train_data[7][0])
# plt.show()

X = []
Y = []

A = []
B = []

C = []
D = []

for features, Labels in train_data:
    X.append(features)
    Y.append(Labels)

for features, labels in eval_data:
    A.append(features)
    B.append(labels)

for features, Labels in test_data:
    C.append(features)
    D.append(Labels)

X_train = X
Y_train = Y

X_Val = A
Y_Val = B

X_test = C
Y_test = D

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_Val = np.array(X_Val)
Y_Val = np.array(Y_Val)

X_test = np.array(X_test)
Y_test = np.array(Y_test)

model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(224,224,3)))
model.add(tf.keras.layers.Conv2D(32, (3,3), activation = tf.keras.activations.relu, padding = 'same'))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(64, (3,3), activation = tf.keras.activations.relu, padding = 'same'))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(128, (3,3), activation = tf.keras.activations.relu, padding = 'same'))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation = tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(3, activation = tf.keras.activations.softmax))      

model.compile(optimizer = tf.keras.optimizers.Adamax(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['Accuracy'])

model.summary()

model.fit(X_train, Y_train, validation_data = (X_Val, Y_Val), epochs = 3)

model.save('pneumonia_detection5.h5')     ##5th save of model at 60% acc

test = tf.keras.models.load_model('pneumonia_detection5.h5')

test.evaluate(X_test, Y_test)