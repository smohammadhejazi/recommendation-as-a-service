from setuptools import setup, find_packages, Extension
from codecs import open
from os import path

from setuptools import dist  # Install numpy right now
dist.Distribution().fetch_build_eggs(['numpy>=1.11.2'])

try:
    import numpy as np
except ImportError:
    exit('Please install numpy>=1.11.2 first.')

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

__version__ = '1.1.1'

here = path.abspath(path.dirname(__file__))

# Get the long description from README.md
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '')
                    for x in all_reqs if x.startswith('git+')]

cmdclass = {}

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [
    Extension(
        'kabirrec.surprise.similarities',
        ['kabirrec/surprise/similarities' + ext],
        include_dirs=[np.get_include()]
    ),
    Extension(
        'kabirrec.surprise.prediction_algorithms.matrix_factorization',
        ['kabirrec/surprise/prediction_algorithms/matrix_factorization' + ext],
        include_dirs=[np.get_include()]),
    Extension('kabirrec.surprise.prediction_algorithms.optimize_baselines',
              ['kabirrec/surprise/prediction_algorithms/optimize_baselines' + ext],
              include_dirs=[np.get_include()]),
    Extension('kabirrec.surprise.prediction_algorithms.slope_one',
              ['kabirrec/surprise/prediction_algorithms/slope_one' + ext],
              include_dirs=[np.get_include()]),
    Extension('kabirrec.surprise.prediction_algorithms.weighted_slope_one',
              ['kabirrec/surprise/prediction_algorithms/weighted_slope_one' + ext],
              include_dirs=[np.get_include()]),
    Extension('kabirrec.surprise.prediction_algorithms.co_clustering',
              ['kabirrec/surprise/prediction_algorithms/co_clustering' + ext],
              include_dirs=[np.get_include()]),
]

if USE_CYTHON:
    ext_modules = cythonize(extensions)
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules = extensions


setup(
    name='kabirrec',
    version='1.0.0',
    description='A recommendation system with cold start, similar items and user specific recommendation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/smohammadhejazi/recommendation-as-a-service',
    author='Seyyed Mohammad Hejazi Hoseini',
    author_email='smohammadhejazi78@gmail.com',
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Education",
    ],
    keywords=['recommendation-service', 'recommendation-system', 'recommender-system'],
    packages=find_packages(),
    install_requires=[
        "joblib>=1.1.0",
        "kmodes>=0.12.1",
        "matplotlib>=3.5.2",
        "matrix_factorization>=1.3",
        "numpy>=1.22.3",
        "pandas>=1.4.2",
        "scikit_learn>=1.1.3",
        "scipy>=1.8.0",
        "six>=1.16.0",
    ]
)
