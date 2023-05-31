# Packages installation

The installation of all packages should be as easy
as go inside each folder and execute the following command:

```python
pip install .
```

It may hapen that you are using `conda` environments and
when you use the `pip` command this one is using the **global**
`pip` program. To solve it, use the correct path of the `pip`
program of your environment. e.g;

```python
/home/wilberquito/.conda/envs/melanoma_api/bin/pip install .
```

To make sure the packages are installed list the pip packages.
For example:

```python
pip list
```

or

```python
/home/wilberquito/.conda/envs/melanoma_api/bin/pip list
```
