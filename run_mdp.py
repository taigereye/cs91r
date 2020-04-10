from mdp_v0 import MdpModelV0
from mdp_v1 import MdpModelV1

import numpy as np

import getopt
import sys


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "m:p:o:")
    except getopt.GetoptError:
        print('usage: run_mdp.py -m <modelversion> -p <paramsfile> -o <outputfile>')
        sys.exit(2)

    version = opts[0][1]
    with open(opts[1][1], 'r') as paramsfile:
        params = eval(paramsfile.read())

    mdp_model = None
    if int(version) == 0:
        mdp_model = MdpModelV0()
    elif int(version) == 1:
        mdp_model = MdpModelV1()

    assert(mdp_model is not None)
    assert(mdp_model.param_names == list(params.keys()))

    mdp_fh = mdp_model.run_single(params)
    stdout_og = sys.stdout
    np.set_printoptions(linewidth=300)

    outfile = open(opts[2][1], 'w')
    sys.stdout = outfile
    mdp_model.print_single(mdp_fh)
    mdp_fh.print_rewards()

    sys.stdout = stdout_og
    outfile.close()


if __name__ == "__main__":
    main(sys.argv[1:])
