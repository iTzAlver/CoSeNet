# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
import setuptools
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

REQUIREMENTS = ['basenet-api>=1.5.3',
                'scikit-learn>=1.1.3',
                'ray>=2.3.0',
                'numpy>=1.24.2',
                'tensorflow>=2.12.0']

setuptools.setup(
    name='cosenet',
    version='1.3.0',
    author='Palomo-Alonso, Alberto',
    author_email='a.palomo@edu.uah',
    description='CoSeNet: AN EXCELLENT APPROACH FOR SEGMENTATION OF CORRELATION MATRICES',
    keywords='deeplearning, ml, api',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/iTzAlver/cosenet.git',
    project_urls={
        'Documentation': 'https://github.com/iTzAlver/cosenet/blob/master/README.md',
        'Bug Reports': 'https://github.com/iTzAlver/cosenet/issues',
        'Source Code': 'https://github.com/iTzAlver/cosenet.git',
        # 'Funding': '',
        # 'Say Thanks!': '',
    },
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    classifiers=[
        # see https://pypi.org/classifiers/
        'Development Status :: 5 - Production/Stable',

        'Topic :: Software Development :: Build Tools',

        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: Apache Software License'
    ],
    python_requires='>=3.8',
    # install_requires=['Pillow'],
    extras_require={
        'dev': ['check-manifest'],
    },
    include_package_data=True,
    install_requires=REQUIREMENTS
)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
