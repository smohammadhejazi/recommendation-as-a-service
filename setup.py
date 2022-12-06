from setuptools import setup, find_packages

setup(
    name='kabirrec',
    version='1.0.0',
    description='A recommendation system with cold start, similar items and user specific recommendation',
    long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
    url='https://github.com/smohammadhejazi/recommendation-as-a-service',
    author='Joshua Lowe',
    author_email='josh@edublocks.org',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Education',
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    keywords=['recommendation-service', 'recommendation-system', 'recommender-system'],
    packages=find_packages(exclude=["devel"]),
    install_requires=[
        "Cython>=0.29.32",
        "flake8>=6.0.0",
        "joblib>=1.1.0",
        "kmodes>=0.12.1",
        "matplotlib>=3.5.2",
        "matrix_factorization>=1.3",
        "numpy>=1.22.3",
        "pandas>=1.4.2",
        "pytest>=7.2.0",
        "scikit_learn>=1.1.3",
        "scipy>=1.8.0",
        "setuptools>=58.1.0",
        "six>=1.16.0",
        "sphinx_rtd_theme>=1.1.1",
        "tabulate>=0.9.0",
    ]
)