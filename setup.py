from setuptools import setup

# read the contents of the README file
from pathlib import Path
this_directory = Path(__name__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='GreenPy',
    version='0.1',
    description='Decimation of the Green function in graphene nanoribbons',
    packages=['GreenPy'],
    zip_safe=False
    )