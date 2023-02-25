from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

BASE_PACKAGE = 'jobshop'

setup(
    name='jobshop',
    version='0.1.0',
    author='Bruno Scalia C. F. Leite',
    author_email='bruscalia12@gmail.com',
    description='Job-shop scheduling problem Python package',
    long_description=long_description,
    packages=find_packages(include=[BASE_PACKAGE, BASE_PACKAGE + '.*']),
    install_requires=[
        "numpy>=1.19.0",
        "pandas",
        "pyomo>=6.0",
        "pymoo==0.6.*",
        "matplotlib",
    ]
)