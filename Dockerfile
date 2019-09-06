FROM python:3
ADD redditCNN_runnable.py /
ADD model.json /
ADD weights.h5 /
RUN pip install pillow praw requests tensorflow keras numpy
ENTRYPOINT ["python", "./redditCNN_runnable.py"]