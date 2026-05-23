from setuptools import setup, find_packages

setup(
    name='datainspector',

    version='0.1.0',

    author='Harendra',

    description='Reusable data cleaning and preprocessing toolkit',

    packages=find_packages(),

    install_requires=[
        'pandas',
        'numpy'
    ],

    python_requires='>=3.8'
)