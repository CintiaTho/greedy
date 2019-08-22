from pathlib import Path

from setuptools import setup, find_packages


# Based on PyPA's sampleproject located at https://github.com/pypa/sampleproject


PROJECT_NAME = 'greedy'


here = Path(__file__).parent
readme_path = here / 'README.md'
version_path = here / PROJECT_NAME / 'VERSION'


with readme_path.open(encoding='utf-8') as r:
    long_description = r.read()


with version_path.open(encoding='utf-8') as v:
    version = v.read()


setup(
    name=PROJECT_NAME,

    version=version,

    description='Collection of Greedy Algorithms',

    long_description=long_description,

    long_description_content_type='text/markdown',

    url='https://github.com/CintiaTho/greedy',

    author='Cintia Lumi Tho',

    author_email='cintia+greedy@tho.net.br',

    classifiers=[
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Education',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'Topic :: Education',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    keywords='greedy graph digraph fractional knapsack kcenter kmeans minimum spanning tree minimum spanning forest prim minimum weight arborescence perceptron precedence task scheduler task scheduler',

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    python_requires='>=3.6, <4',

    install_requires=['numpy', 'graphviz'],

    entry_points={
        'console_scripts': [
            'fk=greedy.core.fk:main',
            'kcenter=greedy.core.kcenter:main',
            'kmeans=greedy.core.kmeans:main',
            'msf_prim=greedy.core.msf_prim:main',
            'mwa=greedy.core.mwa:main',
            'perceptron=greedy.core.perceptron:main',
            'pts=greedy.core.pts:main',
            'ts=greedy.core.ts:main',
        ],
    },

    project_urls={
        'Bug Reports': 'https://github.com/CintiaTho/greedy/issues',
        'Source': 'https://github.com/CintiaTho/greedy/',
    },
)
