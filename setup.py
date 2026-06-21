"""
3D Instruction Ambiguity Detection - Setup Script
==================================================

Installation script for the 3D Instruction Ambiguity Detection package.
"""
from setuptools import setup, find_packages
from pathlib import Path
readme_path = Path(__file__).parent / 'README.md'
long_description = readme_path.read_text(encoding='utf-8') if readme_path.exists() else ''
requirements_path = Path(__file__).parent / 'requirements.txt'
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and (not line.startswith('#'))]
setup(name='ambiver', version='1.0.0', author='Jiayu Ding, Haoran Tang, Hongbo Jin, Wei Gao, Ge Li', description='AmbiVer: Training-Free 3D Instruction Ambiguity Detection', long_description=long_description, long_description_content_type='text/markdown', url='https://github.com/InkMind-AI/Ambiver', packages=find_packages(where='src'), package_dir={'': 'src'}, classifiers=['Development Status :: 4 - Beta', 'Intended Audience :: Developers', 'Intended Audience :: Science/Research', 'License :: OSI Approved :: MIT License', 'Operating System :: OS Independent', 'Programming Language :: Python :: 3', 'Programming Language :: Python :: 3.10', 'Programming Language :: Python :: 3.11', 'Topic :: Scientific/Engineering :: Artificial Intelligence', 'Topic :: Scientific/Engineering :: Image Processing'], python_requires='>=3.10', install_requires=requirements, extras_require={'dev': ['pytest>=7.0.0', 'black>=22.0.0']}, include_package_data=True, zip_safe=False)