import setuptools
from setuptools import find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

BASE_PACKAGE = 'jobshop'

setuptools.setup(
    name='jobshop',
    version='0.0.1.dev3',
    author='DigitalTech',
    author_email='bruscalia12@gmail.com',
    description='Pacote com implementações do jobshop problem',
    long_description=long_description,
    packages=find_packages(include=[BASE_PACKAGE, BASE_PACKAGE + '.*']),
    install_requires=[
        "numpy>=1.19.*",
        "pyomo>=6.*",
        "matplotlib",
    ]
)