# Instructions to lift the server

- Build the image

```sh
sudo docker build -t <image-name> .
```

- Create container from the image

```sh
sudo docker run -t <tag-name> <image-name>
```

- In case you need to `debug` the container
because it turn off inmediatitly after you want to lift it.

```sh
sudo docker run --rm -it <docker-name> /bin/bash
```

To run the project detached use `Podman` or `Docker`:

```sh
sudo podman run -d      \
        --name demo     \
        --rm            \
        -p 8081:8081    \
        <image-name>
```

## The packages

As all this packages were used to train models
and its main funcionality is not in the api
I take them out in the root of the soruce code into
the `_pkgs` folder. Some of the code of this
packages are used in the inference part, so I need
to add them into the image and install them via `pip`.
