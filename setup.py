#!/usr/bin/env python
import pathlib
import setuptools

here = pathlib.Path(__file__).parent.resolve()

setuptools.setup(
    name='w266_project',
    packages=setuptools.find_packages(
        include=['w266_project', 'w266_project.*']
    ),
)
