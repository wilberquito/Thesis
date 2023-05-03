#!bin/bash

SOURCE_CODE_PATH=/home/wilberquito/repos/Thesis/Code

# Make the api image
VICOROBOT_PKG_PATH=$SOURCE_CODE_PATH/vicorobot.pkg
API_PATH=$SOURCE_CODE_PATH/api
API_IMG_NAME=melanoma_api

# Adds vicorobot source code to work with its
# trained models
cp -rf $VICOROBOT_PKG_PATH $API_PATH && \
  cd $API_PATH && \
  docker build -t $API_IMG_NAME .

if [$? -ne 0]; then
  echo "The construction for the API image did not succeed"
  exit 1
fi

# Make the ui image
UI_PATH=$SOURCE_CODE_PATH/ui
UI_IMG_NAME=melanoma_ui

cd $UI_PATH && \
  docker build -t $UI_IMG_NAME .

if [$? -ne 0]; then
  echo "The construction for the UI image did not succeed"
  exit 1
fi

# Download the trained models from registry
PUBLIC_THESIS_URL=https://gitlab.com/wilberquito/open.thesis.git
SAVE_AS=~/open.thesis/

git clone $PUBLIC_THESIS_URL $SAVE_AS

if [$? -ne 0]; then
  echo "Public assets couldn't be downloaded from registry"
  exit 1
fi

# Awake docker services using docker compose and variables
docker compose up

if [$? -ne 0]; then
  echo "Docker services couldn't be awaked"
  exit 1
fi
