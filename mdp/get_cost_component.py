import argparse
import sys

from pathlib import Path

from mdp.models.mdp_V2 import MdpFiniteHorizonV2
from mdp.models.mdp_V3 import MdpFiniteHorizonV3
from mdp.models.mdp_V4 import MdpFiniteHorizonV4


def main(argv):
    parser = argparse.ArgumentParser(description="compute some part of MDP cost")
    parser.add_argument("-m", "--version", help="MDP model version", type=int)
    parser.add_argument("-p", "--paramsfile", help="txt file with args.version specific params dict")
    parser.add_argument("-c", "--component", help="name of desired cost component")
    args = parser.parse_args()

    if int(args.version) < 2:
        print("error: calc_partial_costs only supported for MDP V2 or higher.")
        sys.exit(2)

    params_dir = Path("results/v{}/params".format(args.version))
    pf = params_dir / "p_v{}_{}.txt".format(args.version, args.paramsfile)
    with open(pf, 'r') as paramsfile:
        params = eval(paramsfile.read())
    paramsfile.close()

    mdp_fh = None
    if int(args.version) == 2:
        mdp_fh = MdpFiniteHorizonV2(params)
    elif int(args.version) == 3:
        mdp_fh = MdpFiniteHorizonV3(params)
    elif int(args.version) == 4:
        mdp_fh = MdpFiniteHorizonV4(params)

    assert(mdp_fh is not None)

    costs_dir = Path("results/v{}/costs".format(args.version))
    of = costs_dir / "c_v{}_{}.txt".format(args.version, "{}_{}".format(args.paramsfile, args.component))
    outfile = open(of, 'w+')

    stdout_og = sys.stdout
    sys.stdout = outfile

    mdp_fh.print_partial_costs(args.component)

    sys.stdout = stdout_og
    outfile.close()


if __name__ == "__main__":
    main(sys.argv[1:])
