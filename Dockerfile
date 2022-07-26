FROM tensorflow/tensorflow:2.2.3-py3
RUN apt-get update
RUN python -m pip install --user --upgrade pip
RUN pip3 install --user pytest-shutil
RUN pip3 install --user requests
RUN apt-get install cmake -y
RUN pip3 install --user numpy
RUN pip3 install --user tqdm
RUN pip3 install --user flask
RUN pip3 install --user pymongo
RUN pip3 install --user opencv-python==4.5.5.62
RUN pip3 install --user opencv-contrib-python-headless==4.5.5.62
RUN pip3 install mtcnn-opencv
RUN pip3 install --user keras==2.3.1
RUN pip3 install --user scikit-learn
RUN pip3 install --user imutils
RUN pip3 install Pillow
ADD face-detect  ./