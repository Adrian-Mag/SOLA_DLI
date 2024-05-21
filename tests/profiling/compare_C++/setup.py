# setup.py
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'gaussian',  # The name of the resulting module
        ['gaussian.cpp'],  # Source file
        include_dirs=[pybind11.get_include()],
        language='c++'
    ),
]

setup(
    name='gaussian',
    ext_modules=ext_modules,
    zip_safe=False,
)
