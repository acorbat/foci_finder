from setuptools import setup, find_packages

setup(
    name='foci_finder',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/acorbat/foci_finder/tree/master/',
    license='MIT',
    author='Agustin Corbat',
    author_email='acorbat@df.uba.ar',
    description='Segmenter and analyzer for foci.',
    install_requires=['numpy', 'numba', 'pandas', 'scikit-image', 'scipy',
                      'scikit-learn', 'tifffile', 'trackpy',
                      'cellment @ git+https://github.com/maurosilber/cellment.git',
                      'img_manager @ git+https://github.com/acorbat/img_manager.git',
                      'serialize @ git+https://github.com/hgrecco/serialize.git']
)