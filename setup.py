from setuptools import setup, find_packages

setup(
    name='Canny-Edge-Detector',
    version='0.2',
    packages=find_packages(),
    install_requires=[
        'imageio',
        'scipy',
        'numpy',
        'matplotlib'
    ],
    description="Canny Edge Detector Automation",
    author_email='Dlyalikov01@manhattan.edu',
    author='Dmitri Lyalikov'
)