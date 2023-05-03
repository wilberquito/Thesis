#!bin/bash

WORKING_DIR=$HOME/thesis_deploy

# Creates the path to the working dir
mkdir -p $WORKING_DIR

# Clone private (till the momement) this script
# was created from GitHub
THESIS_URL=git@github.com:wilberquito/Thesis.git
THESIS_SAVE_AS=$WORKING_DIR/thesis/

git clone $THESIS_URL $THESIS_SAVE_AS

SOURCE_CODE_PATH=$WORKING_DIR/thesis/Code

# Make the api image
VICOROBOT_PKG_PATH=$SOURCE_CODE_PATH/vicorobot.pkg
API_PATH=$SOURCE_CODE_PATH/api
API_IMG_NAME=melanoma_api

# Adds vicorobot source code to work with its
# trained models
cp -rf $VICOROBOT_PKG_PATH $API_PATH && \
  cd $API_PATH && \
  docker build -t $API_IMG_NAME .

if [$? -ne 0]
then
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
PUBLIC_THESIS_SAVE_AS=$WORKING_DIR/open.thesis/

git clone $PUBLIC_THESIS_URL $PUBLIC_THESIS_SAVE_AS

if [$? -ne 0]
then
  echo "Public assets couldn't be downloaded from registry"
  exit 1
fi

# Awake docker services using docker compose and variables
docker compose up -d

if [$? -ne 0]
then
  echo "Docker services couldn't be awaked"
  exit 1
fi
