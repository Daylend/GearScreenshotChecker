FROM debian:10

ADD https://raw.githubusercontent.com/Daylend/GearScreenshotChecker/master/redditCNN_runnable.py /
ADD https://raw.githubusercontent.com/Daylend/GearScreenshotChecker/master/model.json /
ADD https://raw.githubusercontent.com/Daylend/GearScreenshotChecker/master/weights.h5 /

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends python3-numpy python3-scipy cython3 python3-h5py python3-grpcio python3-pip python3-setuptools wget -y && \
    pip3 install --no-cache-dir pillow praw requests keras && \
    wget --no-check-certificate -O /tmp/tf_nightly-1.13.1-cp37-none-any.whl https://github.com/mdsimmo/tensorflow-community-wheels/releases/download/1.13.1_cpu_py3_6_amd64/tf_nightly-1.13.1-cp36-cp36m-linux_x86_64.whl && \
    pip3 install --no-cache-dir /tmp/tf_nightly-1.13.1-cp37-none-any.whl && \
    # Cleanup
    apt-get remove wget python3-pip python3-setuptools -y && \
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