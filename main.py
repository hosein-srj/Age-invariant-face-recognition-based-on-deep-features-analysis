import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import imutils
import os
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.models import Model
from keras.models import Sequential
from tensorflow.keras import backend as K


def get_pts(file_pts):
    pts = file_pts.readlines()
    pts = pts[3:len(pts) - 1]
    points = np.zeros((len(pts), 2))
    for i in range(len(pts)):
        pts[i] = pts[i].replace('\n', '')
        x = pts[i].split(' ')
        points[i][0] = float(x[0])
        points[i][1] = float(x[1])

    angle = math.atan2(points[31, 1] - points[36, 1], points[31, 0] - points[36, 0]) * 180 / math.pi
    if angle > 90:
        angle = angle - 180
    if angle < -90:
        angle = angle + 180
    return points, angle


def preprocessing():
    image_path = "FGNET/images"
    pts_path = "FGNET/points"

    for filename in os.listdir(image_path):
        image = cv2.imread(os.path.join(image_path, filename))
        name_pts = filename.split('.')[0].lower() + ".pts"

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        file_pts = open(os.path.join(pts_path, name_pts))

        points, angle = get_pts(file_pts)

        rotated = imutils.rotate(image, angle)
        m_i = min(points[16][1], points[22][1])
        if filename != "033A30.JPG":
            croped = rotated[int(m_i - 10):int(points[7][1]), int(points[0][0] - 10):int(points[14][0] + 10)]
        else:
            croped = rotated[int(m_i - 10):int(points[7][1]), int(points[0][0]):int(points[14][0] + 5)]
        rot_crop_his = cv2.equalizeHist(croped)
        try:
            cv2.imwrite(os.path.join("/content/images_new", filename), rot_crop_his)
        except:
            print(filename)


def get_images_new():
    image_path = "images_new"
    labels = np.zeros((1, 1002))
    i = 0
    images = []
    print(labels.shape)
    for filename in os.listdir(image_path):
        img = load_img(os.path.join(image_path, filename), target_size=(224, 224))
        img = img_to_array(img)
        images.append(img)
        labels[0][i] = int(filename.split('A')[0])
        i = i + 1
    return np.array(images), labels


def feature_extraction():
    model = tf.keras.applications.VGG16(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )

    images, labels = get_images_new()
    f = open('data.txt', 'w')
    l = len(model.layers)
    for i in range(len(images)):
        image = images[i]
        img = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

        keras_function = K.function([model.get_layer(index=0).input], model.get_layer(index=l - 4).output)
        Flatten = keras_function([img])

        keras_function = K.function([model.get_layer(index=0).input], model.get_layer(index=l - 3).output)
        FC7 = keras_function([img])

        keras_function = K.function([model.get_layer(index=0).input], model.get_layer(index=l - 2).output)
        FC6 = keras_function([img])

        f.write(str(labels[0][i]) + '\n')
        np.savetxt(f, Flatten, fmt="%s")
        np.savetxt(f, FC7, fmt="%s")
        np.savetxt(f, FC6, fmt="%s")
    f.close()



if "__main__":
    preprocessing()
    feature_extraction();

