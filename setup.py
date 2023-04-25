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
    entry_points={
        'console_scripts': [
            'your_project_name = your_project_name.main:main'
        ]
    },

    description="Canny Edge Detector Automation",
    author_email='Dlyalikov01@manhattan.edu',
    author='Dmitri Lyalikov'
)