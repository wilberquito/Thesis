from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='modular',
   version='1.0',
   description='Source code used for trainning melanoma models',
   author='wilberquito',
   long_description=long_description,
   author_email='wbq.software@gmail.com',
   packages=find_packages(),
   install_requires=[
        'pandas',
        'numpy',
        'torch',
        'torchvision',
        'opencv_python',
        'albumentations==0.4.6',
        'scikit-learn',
        'mlxtend',
        'wandb'
       ]
)
