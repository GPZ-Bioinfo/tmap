import os
import shutil
import sys
from distutils.command.clean import clean as Clean

from setuptools import setup, find_packages


if sys.version_info < (3, 0):
    sys.exit("tmap can only be used with Python 3. You are currently "
             "running Python %d." % sys.version_info.major)


class CleanCommand(Clean):
    description = "Remove build artifacts from the source tree"

    def run(self):
        Clean.run(self)
        # Remove c files if we are not within a sdist package
        cwd = os.path.abspath(os.path.dirname(__file__))
        remove_c_files = not os.path.exists(os.path.join(cwd, 'PKG-INFO'))
        if remove_c_files:
            print('Will remove generated .c files')
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('sklearn'):
            for filename in filenames:
                if any(filename.endswith(suffix) for suffix in
                       (".so", ".pyd", ".dll", ".pyc")):
                    os.unlink(os.path.join(dirpath, filename))
                    continue
                extension = os.path.splitext(filename)[1]
                if remove_c_files and extension in ['.c', '.cpp']:
                    pyx_file = str.replace(filename, extension, '.pyx')
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == '__pycache__':
                    shutil.rmtree(os.path.join(dirpath, dirname))


cmdclass = {'clean': CleanCommand}

setup(name='tmap',
      version='1.2',
      description='A topological data analysis framework implementing the TDA Mapper algorithm for population-scale microbiome data analysis ',
      author='Haokui Zhou, Tianhua Liao',
      author_email='zhouhaokui@hotmail.com',
      license='GNU',
      url='https://github.com/GPZ-Bioinfo/tmap',
      packages=find_packages(),
      package_data={'': ['test_data/*.csv',
                         'test_data/*.tsv',
                         'example/*'],
                    },
      scripts=['tmap/api/envfit_analysis.py',
               'tmap/api/Network_generator.py',
               'tmap/api/SAFE_analysis.py',
               'tmap/api/SAFE_visualization.py',
               'tmap/api/quick_vis.py'],
      install_requires=['statsmodels>=0.9.0',
                        'tqdm',
                        'scikit-learn>=0.19.1',
                        'matplotlib>=2.2.2',
                        'networkx==2.2',
                        'pandas>=0.23.0',
                        'scipy',
                        'matplotlib!=3.0.0rc2',
                        'umap-learn',
                        'rpy2',
                        'plotly',
                        'numpy'
                        ],
      cmdclass=cmdclass,
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Operating System :: OS Independent",
          "Topic :: Scientific/Engineering :: Bio-Informatics",
      ],
      zip_safe=False,
      )
