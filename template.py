import os

directories = [
    'data/raw',
    'data/processed',
    'notebooks',
    'scripts',
    'models',
    'src',
    'tests',
    'configs',
    'logs',
    'outputs'
]

files = {
    'requirements.txt': '''numpy
pandas
opencv-python
torch
torchvision
matplotlib
scikit-learn
tqdm
hydra-core
omegaconf
''',
    'setup.py': '''from setuptools import setup, find_packages

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
''',
    'README.md': '# EchoNet LV Segmentation\n\nProject for segmenting the left ventricle using the EchoNet-Dynamic dataset.',
    '.gitignore': '*.pyc\n__pycache__/\n.env\nvenv/\n'
}

def create_directories(dirs):
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f'Created directory: {dir_path}')

def create_files(files_dict):
    for file_path, content in files_dict.items():
        with open(file_path, 'w') as f:
            f.write(content)
        print(f'Created file: {file_path}')

def main():
    create_directories(directories)
    create_files(files)
    print('Project structure created successfully in the current directory.')

if __name__ == '__main__':
    main()
