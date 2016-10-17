"""Setup file for mve."""
import os
import sys

from setuptools import setup, find_packages

import ellipsoid

version = ellipsoid.__version__

if sys.argv[-1] == 'publish':
    if os.system("pip freeze | grep wheel"):
        print("wheel not installed.\nUse `pip install wheel`.\nExiting.")
        sys.exit()
    if os.system("pip freeze | grep twine"):
        print("twine not installed.\nUse `pip install twine`.\nExiting.")
        sys.exit()
    os.system("python setup.py sdist bdist_wheel")
    os.system("twine upload dist/*")
    print("You probably want to also tag the version now:")
    print("  git tag -a %s -m 'version %s'" % (version, version))
    print("  git push --tags")
    sys.exit()

setup(
    name='mve',
    version=version,
    description=(
        'Minimum Volume Ellipsoid Analysis'
    ),
    url='https://github.com/JoeErnst/MVE',
    author='Joe Ernst, Philipp Dufter',
    author_email='joe.m.ernst@gmail.com',
    license='BSD',
    packages=find_packages(exclude=['tests*']),
    install_requires=[
        'numpy>=1.9.2',
        'pandas>=0.17.0',
        'scipy>=0.17.0'
    ],
#    test_suite='tests',
#    tests_require=[
#        'coverage>=3.7.1',
#        'nose==1.3.7'
#    ],
    zip_safe=False,
    estimator=[
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
    ]
)
