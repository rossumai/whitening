from os import path
from setuptools import setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='whitening',
      version='0.2',
      description='Document whitening (foreground separation)',
      url='https://github.com/rossumai/whitening',
      author='Bohumir Zamecnik',
      author_email='bohumir.zamecnik@gmail.com',
      license='MIT',
      packages=['whitening'],
      zip_safe=False,
      install_requires=[
          'numpy',
          'Pillow',
          'scikit-image',
          'scipy',
      ],
      extras_require={
          'test': ['pytest'],
      },
      long_description=long_description,
      long_description_content_type='text/markdown',
      # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 3 - Alpha',

          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',

          'License :: OSI Approved :: MIT License',

          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.7',

          'Operating System :: POSIX :: Linux',
      ],
      entry_points={
          'console_scripts': [
              'whiten = whitening.__main__:main'
          ]
      },
      )
