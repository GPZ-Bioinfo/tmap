from setuptools import setup, find_packages

setup(name='tmap',
      version='1.0',
      description='mapper ',
      author='Haokui Zhou',
      author_email='zhouhaokui@hotmail.com',
      license='GNU',
      url='https://github.com/GPZ-Bioinfo/tmap',
      packages=find_packages(),
      package_data={'': ['test_data/*.csv',
                         'test_data/*.tsv',
                         'example/*'],
                    },
      install_requires=['statsmodels>=0.9.0',
                        'tqdm',
                        'scikit-learn>=0.19.1',
                        'matplotlib>=2.2.2',
                        'networkx==1.11',
                        'pandas>=0.23.0',
                        'numpy>=1.10.4',
                        'scipy',
			'matplotlib!=3.0.0rc2',
#                        'scikit-bio>=0.5.2'
                        ],
      zip_safe=False,
      extras_require={'alldeps': ('numpy>=1.10.4', 'scipy',)}
      )
