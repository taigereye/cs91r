import sys

import numpy as np
from pathlib import Path

import mdp.analysis.MdpCLI as cl
import mdp.analysis.mdp_visualize as mv
from mdp.analysis.MdpCLI import MdpArgs


MDP_VERSION = 3
DIR_VERSION = 4


def main(argv):
    parser = MdpArgs(description="extract mean CO2 emissions at intervals of following MDP instance optimal policy")
    parser.add_paramfile_single()
    parser.add_cycle_length()
    parser.add_iterations()
    parser.add_save()
    args = parser.get_args()

    if not args.paramsfile:
        print("error: must pass in paramsfile.")
        sys.exit(2)

    params = cl.get_params_single(MDP_VERSION, args.paramsfile)
    mdp_model = cl.get_mdp_model(MDP_VERSION, [params])
    mdp_fh = cl.get_mdp_instance_single(mdp_model, params)

    t_range = [0, mdp_fh.n_years]

    _, _, y_emit, _ = mv.avg_co2_probabilistic_v(mdp_fh, t_range[0], t_range[1], args.iterations, True)
    y_emit = np.sum(y_emit, axis=0)/args.iterations

    for y in args.cycle:
        targets = []
        # Extract mean CO2 emissions level at intervals of cycle length.
        for i in range(mdp_fh.n_years//y):
            targets.append(y_emit[i])
            i += y
        targets_dir = Path("visuals/v{}/targets".format(DIR_VERSION))
        tf = targets_dir / "e_v{}_{}_{}_mean.txt".format(DIR_VERSION, y, args.paramsfile.replace("co2_tax_", ""))
        with open(tf, 'w+') as targetsfile:
            targetsfile.write(str(targets))
        targetsfile.close()


if __name__ == "__main__":
    main(sys.argv[1:])
