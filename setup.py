from distutils.core import setup

setup(
    name='SmarterThanBackProp',
    version='1.0',
    description='A package containg controlled based methods for training neural networks',
    author='Conrad Li',
    author_email='conradliste@utexas.edu',
    requires=[ 'numpy', 'matplotlib','torch'],
    packages=['utilities', 'learningViz'],
    # package_data={
    # 	'utilities': ['*'],
    # 	'utilities.utils': ['*'],
    # },
)