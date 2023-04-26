from setuptools import setup, find_packages

setup(
    name='pdt-extract',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'imageio',
        'scipy',
        'numpy',
        'matplotlib',
        'circle_fit'
    ],
    url='https://github.com/DmitriLyalikov/pdt-canny-edge-detector',
    entry_points={
        'console_scripts': [
            'pdt-extract = pdt_canny:main'
        ]
    },

    description="Perform edge profile and characteristic feature extraction from images of pendant drops",
    long_description="This package provides modules and automation to perform edge profile extraction and characteristic feature approximation from pendant drop images.",
    long_description_content_type='text/markdown',
    author_email='Dlyalikov01@manhattan.edu',
    author='Dmitri Lyalikov'
)