
# init setup.py
from setuptools import setup, find_packages

# read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# read requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name='pinn_loss',
    version='0.1',
    description='A package for PINN loss with graph neural networks',
    long_description=long_description,
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
)