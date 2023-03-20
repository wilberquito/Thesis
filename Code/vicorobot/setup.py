from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='vicorobot',
   version='1.0',
   description='A module',
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
        'resnest',
        'geffnet',
        'pretrainedmodels',
        'warmup-scheduler'
       ],
)