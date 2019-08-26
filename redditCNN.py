from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, BatchNormalization, Activation
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from keras import backend
import tensorflow as tf
from pathlib import Path
import praw
import praw.exceptions as Exceptions
import requests
from io import BytesIO
from PIL import Image
import os
import sys

imageSize = 128
comment = "Hi there! This post has been removed because it appears to contain a gear screenshot." \
          "\n\nWe receive a high volume of people frequently asking for help on gear progression or grinding spots. " \
          "There are helpful links located at the top of the subreddit, but here are some answers to frequently asked questions:" \
          "\n\n\n[What gear should I go for next?](https://grumpygreen.cricket/bdo-gear-progression-guide/)" \
          "\n\n[Where should I grind?](https://docs.google.com/spreadsheets/d/1gPOFA0uMh_Xc6_pZ_e7wjXRsmkQ3wDuNmhSqiahlvTw/edit#gid=761402636)" \
          "\n\n\nIf you have any further questions or need more help, feel free to leave a comment in the Daily FAQ thread located at the top of the subreddit.  " \
          "\n\n^(Note: I am a bot, and this feature is currently in beta! Sometimes I make mistakes! My last recorded" \
          " accuracy is 97.5% against 350 test examples. If you wish to leave feedback about the bot, feel free to PM)" \
          " /u/Daylend10 ^(or leave us a message in modmail.)"

# Use GPU
config = tf.ConfigProto(device_count={'GPU': 1, 'CPU':4})
sess = tf.Session(config=config)
backend.set_session(sess)

# Initialize CNN
classifier = Sequential()

# Convolution
#classifier.add(Convolution2D(int(imageSize/2), 3, 3, input_shape=(imageSize,imageSize,3), activation='relu'))
#classifier.add(Convolution2D(int(imageSize/4), 3, 1, input_shape=(imageSize, imageSize, 3)))
#classifier.add(BatchNormalization(epsilon=0.001))
classifier.add(Convolution2D(64, kernel_size=(3, 3), activation='relu', input_shape=(imageSize,imageSize, 3)))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(BatchNormalization())

classifier.add(Convolution2D(64, kernel_size=(3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(BatchNormalization())

classifier.add(Convolution2D(64, kernel_size=(3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(BatchNormalization())

classifier.add(Convolution2D(128, kernel_size=(3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(BatchNormalization())

classifier.add(Convolution2D(32, kernel_size=(3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.2))

classifier.add(Flatten())
classifier.add(Dense(128, activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(64, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(1, activation='sigmoid'))

# Pooling
#classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout regularization
#classifier.add(Dropout(0.2))

# Flattening
#classifier.add(Flatten())

# "Full connection" whatever that means
#classifier.add(Dense(units=imageSize*2, activation="relu"))
#classifier.add(Dense(units=1, activation="sigmoid"))

# compile
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# implement early stopping
es = EarlyStopping(monitor="val_acc", mode="auto", patience=100, verbose=1)

mc = ModelCheckpoint("checkpoint.h5", monitor="val_acc", save_best_only=True, mode="max", verbose=1)

# Save model
#model_json = classifier.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)
#print('saved model')

# Optionally load weights, otherwise start training
loadDataAnswer = input("Load saved weights?")
if loadDataAnswer == "n":
    # Rescales, translates, zooms, and flips the image to create more training data
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False)

    # Rescales test data for more test data
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Define training set
    training_set = train_datagen.flow_from_directory(
        'D:\\HDD Storage\\Pictures\\RedditGear\\Train',
        target_size=(imageSize, imageSize),
        batch_size=32,
        class_mode='binary')

    # Define test set
    test_set = train_datagen.flow_from_directory(
        'D:\\HDD Storage\\Pictures\\RedditGear\\Test',
        target_size=(imageSize, imageSize),
        batch_size=32,
        class_mode='binary')

    classifier.fit_generator(
        training_set,
        steps_per_epoch=200,
        epochs=50,
        validation_data=test_set,
        validation_steps=20,
        callbacks=[es, mc])

    classifier.save_weights("weights.h5")
else:
    classifier.load_weights("weights.h5")

#test_set.reset()
#eval = classifier.evaluate_generator(generator=test_set, steps=10, verbose=1)
#print(eval)

"""
while(True):
    try:
        imagePath = Path(input("Image path:"))

        test_image = image.load_img(imagePath, target_size=(imageSize,imageSize))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = classifier.predict(test_image)

        print(result[0][0])
    except:
        pass
"""

# Save predictions to file for viewing
# Stripped down version of redditCNN_runnable for testing
reddit = x
#subreddit = reddit.subreddit('2007scape+shittyrobots+awww+softwaregore+mildlyinteresting+blackdesertonline')
subreddit = reddit.subreddit('blackdesertonline')
#subreddit = reddit.subreddit('testingground4bots')
for submission in subreddit.stream.submissions():
    url = submission.url
    permalink = submission.permalink
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
            tempPath = str(Path("temp/"+filename))
            gearPath = str(Path("gear/"+filename))
            notGearPath = str(Path("notgear/"+filename))

            newimg = open(tempPath, 'wb')
            newimg.write(img_data)
            newimg.close()

            test_image = image.load_img(tempPath, target_size=(imageSize, imageSize))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            result = classifier.predict(test_image)

            if result[0][0] <= 0.1:
                # If the file exists, we've already seen the post, so do nothing
                if not os.path.isfile(gearPath):
                    print(
                        "https://reddit.com" + str(permalink) + " \t\t\t " + str(url) + " \t\t\t " + str(result[0][0]))
                    print(subreddit.display_name)
                    try:
                        os.rename(tempPath, gearPath)
                    except:
                        print("Something went wrong! " + sys.exc_info()[0])
                else:
                    os.remove(tempPath)
            else:
                if not os.path.isfile(notGearPath):
                    os.rename(tempPath, notGearPath)
                    print(
                        "https://reddit.com" + str(permalink) + " \t\t\t " + str(url) + " \t\t\t " + str(result[0][0]))
                else:
                    os.remove(tempPath)




            #answers.write("<image src=\""+url+"\" /><break /><p>"+str(result)+"</p>")
            #print("<image src=\""+str(url)+"\" /><break /><p>"+str(result[0][0])+"</p>", file=open("answers.html", "a"))