import getopt
import sys

from pathlib import Path

from mdp.models.mdp_v0 import MdpModelV0
from mdp.models.mdp_v1 import MdpModelV1
from mdp.models.mdp_v2 import MdpModelV2


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "m:p:o:")
    except getopt.GetoptError:
        print('usage: run_mdp.py -m <modelversion> -p <paramsfile> -o <outputfile>')
        sys.exit(2)

    version = str(opts[0][1])

    params_dir = Path("results/params_v" + version)
    pf = params_dir / opts[1][1]
    with open(pf, 'r') as paramsfile:
        params = eval(paramsfile.read())
    paramsfile.close()

    mdp_model = None
    if int(version) == 0:
        mdp_model = MdpModelV0()
    elif int(version) == 1:
        mdp_model = MdpModelV1()
    elif int(version) == 2:
        mdp_model = MdpModelV2()

    assert(mdp_model is not None)
    assert(mdp_model.param_names == list(params.keys()))
    mdp_fh = mdp_model.run_fh(params)

    runs_dir = Path("results/runs_v" + version)
    of = runs_dir / opts[2][1]
    outfile = open(of, 'w+')

    stdout_og = sys.stdout
    sys.stdout = outfile

    mdp_model.print_fh(mdp_fh)

    sys.stdout = stdout_og
    outfile.close()


if __name__ == "__main__":
    main(sys.argv[1:])
