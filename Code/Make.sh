#!bin/sh

SOURCE_CODE_PATH=~/repos/thesis

# Make the api image
VICOROBOT_PKG_PATH=$SOURCE_CODE_PATH/vicorobot.pkg
API_PATH=$SOURCE_CODE_PATH/api
API_IMG_NAME=melanoma_api

# Adds vicorobot source code to work with its
# trained models
cp -rf $VICOROBOT_PKG_PATH $API_PATH && \
  cd $API_PATH && \
  docker build -t $API_IMG_NAME

# Make the ui image
UI_PATH=$SOURCE_CODE_PATH/ui
UI_IMG_NAME=melanoma_ui

cd $UI_PATH && \
  docker build -t $UI_IMG_NAME

# Download the trained models from registry
PUBLIC_THESIS_URL=https://gitlab.com/wilberquito/open.thesis.gi
SAVE_AS=~/open.thesis/

git lfs clone $PUBLIC_THESIS_URL $SAVE_AS

# Awake docker services using docker compose and variables
docker compose up
