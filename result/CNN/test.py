import numpy as np
import cv2
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.layers import Activation
import tkinter as tk
from tkinter import filedialog
import keras
root = tk.Tk()
root.withdraw()

img_width, img_height = 64,64
input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(256, (3, 3), input_shape = input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size =(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size =(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(15, activation='softmax'))
model.summary()

opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss ='categorical_crossentropy',optimizer=opt, metrics =['accuracy'])

model.load_weights('model_saved.h5')
print("done loading weights of the trained model")

labels = ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___healthy','Corn_(maize)___Northern_Leaf_Blight',
                'Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___healthy','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Potato___Early_blight','Potato___healthy','Potato___Late_blight','Tomato__Tomato_YellowLeaf__Curl_Virus',
                'Tomato_Early_blight','Tomato_healthy','Tomato_Late_blight']

# input image
file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
img = cv2.imread(file_path)

imgout = img.copy()

##====================Classification================================
img1 = cv2.resize(img, (64, 64))
img1 = img1 / 255
img = np.reshape(img1, [1, 64, 64, 3])

# predict
classes = model.predict(img)
print(classes)
output = np.argmax(classes)
print(np.argmax(classes))
print(labels[output])

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(imgout, labels[output], (30, 30), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
cv2.imwrite('output.jpg', imgout)
cv2.imshow('output', imgout)
cv2.waitKey(0)
