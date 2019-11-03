# -*- coding: utf-8 -*-
import io
import os
import re
from setuptools import setup, find_packages


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


setup(
    name="serket",
    version="0.0.1",
    url="https://github.com/naka-lab/Serket",

    # author="",
    # author_email="",

    description="Symbol Emergence in Robotics tool KIT",
    long_description=read("README.md"),

    packages=find_packages(exclude=('tests',)),

    install_requires=[],

    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
