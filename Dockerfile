FROM alpine:3.10
RUN apk add --no-cache python3
ADD redditCNN_runnable.py /
ADD model.json /
ADD weights.h5 /

#RUN package installer if you need more dependant softare for your app
#apk add --no-cache bash nodejs shadow #etc #etc

RUN pip3 install pillow praw requests tensorflow keras numpy

ENV clientid=id \
    clientsecret=secret \
    username=username \
    password=password

CMD python3 redditCNN_runnable.py