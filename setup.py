from setuptools import setup, find_packages

# Load the contents of your README file
with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='sola',
    version='0.1',
    author='Your Name',
    author_email='your_email@example.com',
    description='A brief description of your package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Adrian-Mag/SOLA_DLI.git',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.17',
        'networkx',
        'matplotlib',
        'plotly'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
