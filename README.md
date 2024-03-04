# A platform for classifying melanoma

This repository contains the source code of my master's thesis in Data Science,
I hardly recommend you to [read
it](https://raw.githubusercontent.com/wilberquito/melanoma.thesis/remake/Doc/remake/main.pdf).
The assets and the trained models can be found in this public GitLab
[repository](https://gitlab.com/wilberquito/open.thesis).

## Manual Installation

We are pleased to offer you all the necessary tools required to effortlessly
set up the CAD infrastructure. Managing the installation of Python and NPM
packages is something you need not worry about, as we have already created
comprehensive images that encompass all the essential packages, ensuring a
seamless execution process.

It's important to note that the tools needed for the installation are primarily
focused on project building and remain independent of the services created.
Although our example uses Fedora Linux 38 as the operating system, rest assured
that all the tools used in this platform are cross-platform compatible,
guaranteeing smooth execution on other operating systems as well.

Here is the list of tools needed for the installation:

- Git
- Docker or Podman
- docker-compose
- cURL

Before proceeding, please make sure you have all the required tools listed
above. Next, you'll need to become a superuser, and you can do that with the
following command:

```bash
$ sudo su
```

If you opt for Docker, execute the following command to initiate the
installation process:

```bash
$ curl https://raw.githubusercontent.com/wilberquito/melanoma.thesis/main/MAKE.sh | bash
```

If you prefer Podman, you will first need to install the podman docker package
to enable Podman to work harmoniously with Docker commands:

```bash
$ sudo dnf install -y podman-docker
```

Afterward, you can utilize the MAKE.sh script to build the architecture:

```bash
$ curl https://raw.githubusercontent.com/wilberquito/melanoma.thesis/main/MAKE.sh | bash
```

To confirm that the CAD infrastructure services are operational, you can locate
the two containers using the following command:

```bash
$ docker ps
```

You should see the two containers.

The API is accessible through port 8082. To view the documentation, simply open
any web browser and enter the following URL:

```bash
http://127.0.0.1:8082/docs
```

Similarly, to access the CAD UI, you can use port 5174. Just open your web
browser and enter the following URL:

```bash
http://127.0.0.1:5174
```

Rest assured that with these tools and guidelines, you'll have a smooth and
successful installation of the CAD infrastructure.
