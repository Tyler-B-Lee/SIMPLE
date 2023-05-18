from setuptools import setup, find_packages

setup(
    name='minecraftcg',
    version='0.3.0',
    description='Minecraft Card Game? Gym Environment, V3',
    packages=find_packages(),
    install_requires=[
        'gym>=0.9.4,<=0.15.7',
        'numpy>=1.13.0',
        'opencv-python>=3.4.2.0',
    ]
)


