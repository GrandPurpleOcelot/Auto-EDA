import setuptools
import os

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = [] 
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="majora", # Replace with your own username
    version="0.0.1",
    author="Thien Nghiem",
    author_email="thien.nghiem94@gmail.com",
    description="Majora is a python library that automates common tasks in your exploratory data analysis. ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GrandPurpleOcelot/Auto-EDA",
    packages=setuptools.find_packages(),
    install_requires=install_requires
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)