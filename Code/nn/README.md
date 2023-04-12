# NN package Melanoma classifier

In this file I'll explain to myself which are the instructions to make this run.

## Set up

To create environment from `environment.yml`

```sh
$ conda create --file=environment.yml
```

To update the environment:

```sh
$ conda env update -n <env-name> --file environment.yml
```

To remove the environment:

```sh
$ conda env remove -n <env-name>
```

## The data

You need to provide the data as given by the `downloader.sh`, but you need to drop images
and entries of each csv of classes like:

- Unknown
- cafe-au-lait macule
- atypical melanocytic proliferation

To make it possible, I created the `etl.py` file that makes the work for you.
