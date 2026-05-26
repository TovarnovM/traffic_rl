FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel
ENV TZ=Europe/Moscow \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    bash ffmpeg libsm6 libxext6 \
    build-essential python3-dev \
 && rm -rf /var/lib/apt/lists/*


ARG USERNAME=dev
ARG UID=1000
ARG GID=1000

RUN groupadd --gid $GID $USERNAME && \
    useradd --uid $UID --gid $GID -m $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN chsh -s /bin/bash root
RUN chsh -s /bin/bash $USERNAME

# RUN mkdir /workspace
COPY .devcontainer/requirements.txt .

RUN pip3 install -r requirements.txt

COPY .devcontainer/requirements.dev.txt .

RUN pip3 install -r requirements.dev.txt

USER $USERNAME

WORKDIR /workspace
ENV PYTHONPATH=/workspace/src:${PYTHONPATH}
# Копируем список зависимостей



# USER ${USERNAME}