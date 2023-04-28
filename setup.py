from setuptools import setup, find_packages

setup(
    name='pdt-extract',
    version='0.1.3',
    packages=['pdt_extract'],
    install_requires=[
        'imageio',
        'scipy',
        'numpy',
        'matplotlib',
        'circle_fit',
        'pandas'
    ],
    classifiers=[
        'Intended Audience :: Science/Research'
    ],
    url='https://github.com/DmitriLyalikov/pdt-canny-edge-detector',

    description="Perform edge profile and characteristic feature extraction from images of pendant drops",
    long_description="This package provides modules and automation to perform edge profile extraction and characteristic feature approximation from pendant drop images.",
    long_description_content_type='text/markdown',
    author_email='Dlyalikov01@manhattan.edu',
    author='Dmitri Lyalikov'
)