FROM ubuntu:18.04
RUN apt-get update 
RUN apt-get upgrade -y
RUN apt install -y software-properties-common
RUN apt-get install --reinstall ca-certificates
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.9
RUN apt install -y python3.9-distutils
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade setuptools
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade distlib
WORKDIR /workspace
ADD . /workspace/
ENV HOME=/workspace
RUN pip install -r requirements.txt
CMD ["python", "execute_GUI.py"]
