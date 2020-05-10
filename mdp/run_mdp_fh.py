import argparse
import sys

from pathlib import Path

from mdp.models.mdp_v0 import MdpModelV0
from mdp.models.mdp_v1 import MdpModelV1
from mdp.models.mdp_v2 import MdpModelV2
from mdp.models.mdp_v3 import MdpModelV3


def main(argv):
    parser = argparse.ArgumentParser(description="run MDP instance")
    parser.add_argument("-m", "--version", help="MDP model version", type=int)
    parser.add_argument("-p", "--paramsfile", help="txt file with version specific params dict")
    args = parser.parse_args()

    params_dir = Path("results/v{}/params".format(args.version))
    pf = params_dir / "p_v{}_{}.txt".format(args.version, args.paramsfile)
    with open(pf, 'r') as paramsfile:
        params = eval(paramsfile.read())
    paramsfile.close()

    mdp_model = None
    if args.version == 0:
        mdp_model = MdpModelV0()
    elif args.version == 1:
        mdp_model = MdpModelV1()
    elif args.version == 2:
        mdp_model = MdpModelV2()
    elif args.version == 3:
        mdp_model = MdpModelV3()

    assert(mdp_model is not None)
    assert(mdp_model.param_names == list(params.keys()))
    mdp_fh = mdp_model.run_fh(params)

    runs_dir = Path("results/v{}/runs".format(args.version))
    of = runs_dir / "r_v{}_{}.txt".format(args.version, args.paramsfile)
    outfile = open(of, 'w+')

    stdout_og = sys.stdout
    sys.stdout = outfile

    mdp_model.print_fh(mdp_fh)

    sys.stdout = stdout_og
    outfile.close()


if __name__ == "__main__":
    main(sys.argv[1:])
