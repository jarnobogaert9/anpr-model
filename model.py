import os
import cv2
import random
import numpy as np
from sklearn.model_selection  import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential

# Modify to your local structure
images_path = '../ANPR_augmentation/images/Augmented'

categories = os.listdir(images_path)
train_data = []


def create_data():
  for category in categories:
    p = f'{images_path}/{category}'
    label = categories.index(category)
    for img in os.listdir(p):
      # print(img)
      images_array = cv2.imread(f'{p}/{img}', cv2.IMREAD_GRAYSCALE) # 2d array
      # new_arr = np.asarray(images_array)
      # f = np.asarray(new_arr).shape[0]
      # s = np.asarray(new_arr).shape[1]
      # new_arr = new_arr.reshape(f, s, 1)
      # print(new_arr.shape)
      train_data.append([images_array, label])

      # print(images_array)
      # images_array = cv2.imread(f'{p}/{img}')
  # print(categories)
  # print(images_array.shape)
  # print(images_array)

# Is not shuffled
create_data()

# Shuffle data
random.shuffle(train_data)

# for item in train_data:
#   print(item[1])


# Create train/test set (X,y)
x = [feature for feature, label in train_data]
y = [label for feature, label in train_data]

# x has to be a 2d array in order to use it to train our model
# so we reshape it from 1d array to 2d array
# x = np.array(x)
x = np.array(x).reshape(-1, 244, 244, 1)
y = np.array(y)

print("---------------------", x.shape)


x /= 255.00


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

# print(x_train.shape)
# print(type(x_train))
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

# print(x_train)

combined_model = Sequential()

mn_model = MobileNetV2()

# Create layer with output from existing model
# dense_layer = Dense(2, activation='relu')(mn_model.output)
dense_layer = Dense(2, activation='relu')

# Combine the mobile net model with new custom layer
# combined_model = Model(mn_model.input, dense_layer)
combined_model.add(mn_model)
combined_model.add(dense_layer)

for layer in combined_model.layers[:-1]:
  layer.trainable = False

combined_model.summary()

combined_model.compile(optimizer='adam', loss='binary_crossentropy')

combined_model.fit(x_train, y_train)

