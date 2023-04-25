# Instructions to lift the server

- Build the image

```sh
sudo docker build -t <image-name> .
```

- Create container from the image

```sh
sudo docker run -t <tag-name> <image-name>
```

- In case you need to `debub` the container
because it turn off inmediatitly after you want to lift it.

```sh
sudo docker run --rm -it <docker-name> /bin/bash
```

To run the project detached use `Podman` or `Docker`:

```sh
sudo podman run -d \
--name demo \
--rm \
-p 8081:8081 \
<image-name>
```
