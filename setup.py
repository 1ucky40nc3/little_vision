from setuptools import setup
from setuptools import find_packages


setup(
    name='little_vision',
    version='0.1.0',
    packages=find_packages(
        include=[
            'little_vision', 
            'little_vision.*'
        ],
    ),
)