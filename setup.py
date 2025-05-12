from setuptools import setup, find_packages

setup(
    name='echonet',
    version='0.1.0',
    description='Left ventricle segmentation using EchoNet-Dynamic dataset',
    author='Ernesto Serize',
    author_email='ernestoserize@gmail.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'opencv-python',
        'torch',
        'torchvision',
        'matplotlib',
        'scikit-learn',
        'tqdm',
        'hydra-core',
        'omegaconf',
        'matplotlib',
        'seaborn',
    ],
    entry_points={
        'console_scripts': [
            'train=src.models.train:main',
            'evaluate=src.models.evaluate:main',
        ],
    },
)
