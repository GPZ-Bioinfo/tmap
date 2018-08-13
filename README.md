# What is *tmap*?

For large scale and integrative microbiome research, it is expected to apply advanced data mining techniques in microbiome data analysis.

Topological data analysis (TDA) provides a promising technique for analyzing large scale complex data. The most popular *Mapper* algorithm is effective in distilling data-shape from high dimensional data, and provides a compressive network representation for pattern discovery and statistical analysis.

***tmap*** is a topological data analysis framework implementing the TDA *Mapper* algorithm for population-scale microbiome data analysis. We developed ***tmap*** to enable easy adoption of TDA in microbiome data analysis pipeline, providing network-based statistical methods for enterotype analysis, driver species identification, and microbiome-wide association analysis of host meta-data.

# How to Install *tmap*?

To install tmap, run:
```python
git clone https://github.com/GPZ-Bioinfo/tmap.git
cd tmap
python setup.py install
```
If you encounter any error like `Import error: tkinter`, you need to run `sudo apt install python-tk` or `sudo apt install python3-tk`.

# *tmap* Documentation

* [Basic Usage of tmap](https://tmap.readthedocs.io/en/latest/basic.html)
* [How to Choose Parameters in tmap](https://tmap.readthedocs.io/en/latest/param.html)
* [Visualizing and Exploring TDA Network](https://tmap.readthedocs.io/en/latest/vis.html)
* [Network Statistical Analysis in tmap](https://tmap.readthedocs.io/en/latest/statistical.html)
* [How tmap work](https://tmap.readthedocs.io/en/latest/how2work.html)
* [Microbiome Examples](https://tmap.readthedocs.io/en/latest/example.html)
* [API](https://tmap.readthedocs.io/en/latest/api.html)
* [Reference](https://tmap.readthedocs.io/en/latest/reference.html)
* [FAQ](https://tmap.readthedocs.io/en/latest/FAQ.html)

# *tmap* Quick Guides

You can read the [Basic Usage of tmap](https://tmap.readthedocs.io/en/latest/basic.html) for general use of tmap.
Or follow the [Microbiome examples](https://tmap.readthedocs.io/en/latest/example.html) for using tmap in microbiome analysis.

# *tmap* Publication

# Contact Us
If you have any questions or suggestions, you are welcome to contact us via email: haokui.zhou@gmail.com.