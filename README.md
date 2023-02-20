# Vision project

## The api

To create a new environment from the current source code you need to make use of the `environment.yml` file that
contains all the needed dependencies.

```sh
cd Code/API \
conda env create --name <env_name> --file=environment.yml \
conda activate <env_name>
```
