#!/usr/bin/python3

from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, BatchNormalization, Activation
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
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
import getopt


def main(argv):
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

    helpstring = "redditCNN_runnable.py -ci <clientid> -cs <clientsecret> -u <username> -p <password>"

    clientid = ''
    clientsecret = ''
    username = ''
    password = ''

    try:
        opts, args = getopt.getopt(argv,"i:s:u:p:", ["clientid=","clientsecret=","username=","password="])
    except getopt.GetoptError:
        print(helpstring)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--clientid"):
            clientid = arg
        elif opt in ("-s", "--clientsecret"):
            clientsecret = arg
        elif opt in ("-u", "--username"):
            username = arg
        elif opt in ("-p", "--password"):
            password = arg


    # Use GPU
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(device_count={'GPU': 1, 'CPU':4}, gpu_options=gpu_options)
    sess = tf.Session(config=config)
    backend.set_session(sess)

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    classifier = model_from_json(loaded_model_json)
    classifier.load_weights("weights.h5")

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

    print("Checking Reddit now")


    reddit = praw.Reddit(user_agent='Gear Detector (by /u/daylend10)',
                         client_id=clientid, client_secret=clientsecret,
                         username=username, password=password)
    subreddit = reddit.subreddit('blackdesertonline')
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

                if result[0][0] == 0.0:
                    # If the file exists, we've already seen the post, so do nothing
                    if not os.path.isfile(gearPath):
                        print(
                            "https://reddit.com" + str(permalink) + " \t\t\t " + str(url) + " \t\t\t " + str(result[0][0]))
                        try:
                            # We'll just assume that approved posts shouldn't be removed
                            if not submission.approved:
                                reply = submission.reply(comment)
                                reply.mod.distinguish(how='yes', sticky=True)
                                submission.mod.remove(spam=False)
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

if __name__ == "__main__":
    main(sys.argv[1:])