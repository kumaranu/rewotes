from setuptools import setup, find_packages

setup(
    name='kumaranu',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pytest',
        'ase',
        'pandas',
    ],
    entry_points={
        'console_scripts': [],
    },
)
