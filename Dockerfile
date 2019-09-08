FROM debian:10

ADD https://raw.githubusercontent.com/Daylend/GearScreenshotChecker/master/redditCNN_runnable.py /
ADD https://raw.githubusercontent.com/Daylend/GearScreenshotChecker/master/model.json /
ADD https://raw.githubusercontent.com/Daylend/GearScreenshotChecker/master/weights.h5 /

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends python3-numpy python3-scipy cython3 python3-h5py python3-grpcio python3-pip python3-setuptools -y && \
    pip3 install --no-cache-dir pillow praw requests keras tensorflow && \
    # Cleanup
    apt-get remove python3-pip python3-setuptools -y && \
    apt-get autoremove -y && \
    apt-get autoclean -y && \
    apt-get clean -y && \
    apt-get purge -y && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV clientid=id \
clientsecret=secret \
username=username \
password=password

CMD python3 redditCNN_runnable.py 