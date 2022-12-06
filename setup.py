from setuptools import setup, find_packages

setup(
    name='kabirrec',
    version='1.0.1',
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
