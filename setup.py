from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()


setup(
    name='cube-plot',
    version='0.1',
    description='Plot cube files',
    author='Yanze Wu',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
    install_requires=[
        'numpy',
        'plotly >= 5.0',
        'scikit-image',
    ]
)