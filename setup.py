from setuptools import setup, find_packages

setup(
    name='metalsitenn',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'biopython',
        'pandas',
        'numpy',
        'torch'
    ],
    author='Evan Komp',
    author_email='evan.komp@nrel.gov',
    description='Neural network modeling of metal binding sites',
    url='https://github.com/evankomp/metalsitenn',
)