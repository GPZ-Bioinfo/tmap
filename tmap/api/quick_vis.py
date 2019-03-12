import argparse
import pickle

from tmap.api.general import *
from tmap.tda.plot import vis_progressX


def main(args):
    graph = pickle.load(open(args.graph, 'rb'))
    accessory_objs = graph['accessory_obj']
    projected_X = accessory_objs['raw_X']
    vis_progressX(graph, projected_X=projected_X,
                  mode='file',
                  simple=True,
                  filename=args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-G", "--graph", help="Graph file computed from 'Network_generator.py'.",
                        type=str, required=True)
    parser.add_argument("-O", "--output", help="Prefix of output file",
                        type=str, required=True)
    args = parser.parse_args()

    process_output(output=args.output)

    main(args)
