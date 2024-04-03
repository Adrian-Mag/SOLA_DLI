from setuptools import setup, find_packages

# Load the contents of your README file
with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='sola',
    version='0.1',
    author='Adrian Marin Mag',
    author_email='marin.mag@stx.ox.ac.uk',
    description='A brief description of your package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Adrian-Mag/SOLA_DLI.git',
    packages=find_packages(exclude=['tests*']),
    package_data={
        'sola': ['kernels_modeplotaat_Adrian/*'],
    },
    install_requires=[
        'numpy>=1.17',
        'networkx',
        'matplotlib',
        'plotly',
        'scipy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
