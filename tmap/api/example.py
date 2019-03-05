from tmap.api.generate_network import *
from sklearn import datasets

X, y = datasets.make_circles(n_samples=5000, noise=0.05, factor=0.3)

if __name__ == '__main__':
    main(X,'~/test.graph')