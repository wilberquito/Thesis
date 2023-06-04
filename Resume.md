# Resume

- Made 2 python (trainbble code) packages (vicorobot, modular)
- Made the bridge between the packages and the api
- Notebook of data exploration
- Glasary or index of trainned resnet18
- Train different kinds of resnet18 with (show them the notebooks):
 - Different scheduler
 - Every network has exported its trainned information
 - Exported to .tar files
- Each model takes 2 hores and few minutes to finish
thanks to some friends they let me they udg account and I could train
with differnt accounts at the same time
- Create the docker images of api and ui
- Create the dockercompose to handle them at the same time
- Exported the static content and trianed models to my gitlab account
because github don't allow me to work freely with git lfs
- Create a shell script to download the source code (github)
and models and configuration from (gitlab). The script creates
the images, runs the dockercompose file to create the containers using
the the configurations from the .env file
- The containers uses volumes for update models from outsides
and avoid you the create again a new container to do that
it also let you modify the conf files from outside
- Style app
- Select multiple models from the app, handle the state of the interactuables buttons and scrolls
- Sort images by importance or name depending on the state
- Notification system (ui)
