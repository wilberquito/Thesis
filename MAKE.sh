#!bin/bash

WORKING_DIR=$HOME/thesis_deploy

# Creates the path to the working dir
mkdir -p $WORKING_DIR

# Clone private (till the momement) this script
# was created from GitHub
SOURCE_CODE_REPOSITORY=https://github.com/wilberquito/melanoma.thesis.git
SOURCE_CODE_REPOSITORY_SAVE_AS=$WORKING_DIR/thesis/

# Cloning the source code
git clone --depth 1 --branch main $SOURCE_CODE_REPOSITORY $SOURCE_CODE_REPOSITORY_SAVE_AS

SOURCE_CODE_PATH=$WORKING_DIR/thesis/Code
PACKAGES_PATH=$SOURCE_CODE_PATH/_pkgs/

# Make the api (fastapi) image
API_PATH=$SOURCE_CODE_PATH/fastapi
API_IMG_NAME=melanoma_api

# Adds packages into the api source code directory
cp -rf $PACKAGES_PATH $API_PATH && \
  cd $API_PATH && \
  docker build -t $API_IMG_NAME .

if [ $? -ne 0 ]
then
  echo "The construction for the API image did not succeed"
  exit 1
fi

# Make the ui (svelte) image
UI_PATH=$SOURCE_CODE_PATH/svelte
UI_IMG_NAME=melanoma_ui


cd $UI_PATH && \
  docker build -t $UI_IMG_NAME .

if [ $? -ne 0 ]
then
  echo "The construction for the UI image did not succeed"
  exit 1
fi

# Download the trained models from registry
ASSETS_REPOSITORY=https://gitlab.com/wilberquito/open.thesis.git
ASSETS_REPOSITORY_SAVE_AS=$WORKING_DIR/open.thesis

# Cloning the models and the configurations
git clone --depth 1 --branch main $ASSETS_REPOSITORY $ASSETS_REPOSITORY_SAVE_AS

if [ $? -ne 0 ]
then
  echo "Public assets couldn't be downloaded from registry"
  exit 1
fi

# You can change this for any conf environment
# This copies the environment dockercompose and
# env file of the environment dockercompose
CONF_ENVS=$ASSETS_REPOSITORY_SAVE_AS/confs
ENV=demo
ENV_PATH=$CONF_ENVS/$ENV

cp $ENV_PATH/.env $ENV_PATH/docker-compose.yml \
    $SOURCE_CODE_REPOSITORY_SAVE_AS/Code

if [ $? -ne 0 ]
then
  echo "Couldn't copy the - $ENV - files into the thesis folder"
  exit 1
else
  echo "Env - $ENV - copied in the thesis folder"
fi

# Awake docker services using docker compose and variables
cd $SOURCE_CODE_REPOSITORY_SAVE_AS/Code && \
  docker-compose up -d

if [ $? -ne 0 ]
then
  echo "Docker services couldn't be awaked"
  exit 1
fi
