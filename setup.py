from setuptools import setup

setup(name='trainRack',
      version='0.1',
      description='Base module for training models on pyTorch',
      url='http://github.com/VSainteuf/trainRack',
      author='VSainteuf',
      license='MIT',
      packages=['trainRack'],
      zip_safe=False,
      install_requires=['torch','torchnet','numpy'])