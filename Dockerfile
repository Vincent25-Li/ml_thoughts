FROM python:3.9

# Setup Timezone
ENV TZ=Asia/Taipei
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ /etc/Timezone

LABEL authors='Vincent'

# setup User
RUN useradd -m -d /home/vincent -s /bin/bash vincent
USER vincent
WORKDIR /home/vincent

# install packages
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install jupyterlab

# Add jupyterlab path
ENV PATH=/home/vincent/.local/bin:$PATH