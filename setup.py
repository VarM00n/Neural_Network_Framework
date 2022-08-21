from setuptools import setup, find_packages

VERSION = '1.17'
DESCRIPTION = 'Neural network framework'
LONG_DESCRIPTION = 'Neural network framework for learning purposes'

setup(
    name="VMNeuralNetwork",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Var Moon",
    author_email="bbala1351998@gmail.com",
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy', 
                      'scipy'],
    keywords='conversion',
    classifiers= [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
    ]
)
