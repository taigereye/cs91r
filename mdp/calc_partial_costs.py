import getopt
import sys

from pathlib import Path

from mdp.models.mdp_v2 import MdpFiniteHorizonV2


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "m:p:o:c:")
    except getopt.GetoptError:
        print('usage: calc_partial_costs.py -m <modelversion> -p <paramsfile> -o <outputfile> -c <component>')
        sys.exit(1)

    version = str(opts[0][1])
    if int(version) < 2:
        print("error: calc_partial_costs only supported for MDP v2 or higher.")
        sys.exit(2)

    component = str(opts[3][1])

    params_dir = Path("results/params_v" + version + "/")
    pf = params_dir / opts[1][1]
    with open(pf, 'r') as paramsfile:
        params = eval(paramsfile.read())
    paramsfile.close()

    mdp_fh = None
    if int(version) == 2:
        mdp_fh = MdpFiniteHorizonV2(params)

    assert(mdp_fh is not None)

    stdout_og = sys.stdout
    costs_dir = Path("results/costs_v" + version + "/")
    of = costs_dir / opts[2][1]
    outfile = open(of, 'w+')
    sys.stdout = outfile
    mdp_fh.print_partial_costs(component)
    sys.stdout = stdout_og
    outfile.close()


if __name__ == "__main__":
    main(sys.argv[1:])
