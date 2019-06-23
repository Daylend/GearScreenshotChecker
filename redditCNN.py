from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from keras import backend
import tensorflow as tf
from pathlib import Path
import praw
import requests
from io import BytesIO
from PIL import Image
import os

# Use GPU
config = tf.ConfigProto(device_count={'GPU': 1, 'CPU':4})
sess = tf.Session(config=config)
backend.set_session(sess)

# Initialize CNN
classifier = Sequential()

# Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape= (64,64,3), activation= 'relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size= (2,2)))

# Flattening
classifier.add(Flatten())

# "Full connection" whatever that means
classifier.add(Dense(output_dim = 128, activation="relu"))
classifier.add(Dense(output_dim = 1, activation="sigmoid"))

# compile
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Rescales, translates, zooms, and flips the image to create more training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False)

# Rescales test data for more test data
test_datagen = ImageDataGenerator(rescale=1./255)

# Define training set
training_set = train_datagen.flow_from_directory(
    'D:\\HDD Storage\\Pictures\\RedditGear\\Train',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary')

# Define test set
test_set = train_datagen.flow_from_directory(
    'D:\\HDD Storage\\Pictures\\RedditGear\\Test',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary')

# Optionally load weights, otherwise start training
loadDataAnswer = input("Load saved weights?")
if(loadDataAnswer == "n"):
    classifier.fit_generator(
        training_set,
        steps_per_epoch=300,
        epochs=1,
        validation_data=test_set,
        validation_steps=800)

    classifier.save_weights("weights.h5")
else:
    classifier.load_weights("weights.h5")

"""while(True):
    try:
        imagePath = Path(input("Image path:"))

        test_image = image.load_img(imagePath, target_size=(64,64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = classifier.predict(test_image)

        print(result[0][0])
    except:
        pass
"""

# Save predictions to file for viewing
#answers = open("answers.html", "w")

reddit = praw.Reddit(user_agent="Gear Screenshot Detector by /u/Daylend10",
                     client_id="haha", client_secret="nonono",
                     username="ha", password="no")
subreddit = reddit.subreddit('blackdesertonline')
for submission in subreddit.stream.submissions():
    url = submission.url
    if url.endswith(".png") or url.endswith(".jpg") or url.endswith(".jpeg"):
        filename = url.rsplit('/', 1)[1]
        h = requests.head(url, allow_redirects=True)
        header = h.headers

        # Make sure the file size is fairly small (20 mb)
        content_length = header.get('content-length', None)

        if content_length and int(content_length) < 2e7:
            img_data = requests.get(url).content
            #test_image = Image.open(BytesIO(img_data))
            # Havent figured out how to read from memory yet so rip hard drives if you use this on a busy sub
            if not os.path.exists("temp"):
                os.mkdir("temp")
            if not os.path.exists("gear"):
                os.mkdir("gear")
            if not os.path.exists("notgear"):
                os.mkdir("notgear")
            tempPath = Path("temp\\"+filename)
            gearPath = Path("gear\\"+filename)
            notGearPath = Path("notgear\\"+filename)

            newimg = open(tempPath, 'wb')
            newimg.write(img_data)
            newimg.close()

            test_image = image.load_img(tempPath, target_size=(64,64))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = classifier.predict(test_image)

            if result[0][0] <= 0.01:
                os.rename(tempPath, gearPath)
            else:
                os.rename(tempPath, notGearPath)


            print(str(url)+" ---------------------- "+str(result[0][0]))

            #answers.write("<image src=\""+url+"\" /><break /><p>"+str(result)+"</p>")
            print("<image src=\""+str(url)+"\" /><break /><p>"+str(result[0][0])+"</p>", file=open("answers.html", "a"))

