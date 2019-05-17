import os
from setuptools import setup, find_packages


def read(file_name):
    return open(os.path.join(os.path.dirname(__file__), file_name)).read()


setup(
    name='Pattern Assocaiation',
    version='0.1',
    url='',
    license='GNU Affero General Public License v3.0',
    author='Trevor Xander',
    author_email='trevorcolexander@gmail.com',
    long_description=read('README.md'),
    install_requires=['pandas', 'numpy', 'jupyter', 'xlrd']
)
