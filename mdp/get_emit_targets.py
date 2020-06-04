import sys

import numpy as np
from pathlib import Path

import mdp.analysis.MdpCLI as cl
import mdp.analysis.MdpData as mv
from mdp.analysis.MdpCLI import MdpArgs


MDP_VERSION = 3
DIR_VERSION = 4


def calc_emit_dec(mdp_fh, y_emit, cycle):
    emit_t0, emit_tN = y_emit[0], y_emit[mdp_fh.n_years-1]
    emit_dec = round((emit_t0-emit_tN) / (mdp_fh.n_years//cycle-1), 2)
    return emit_dec


def main(argv):
    parser = MdpArgs(description="extract mean CO2 emissions at intervals of following MDP instance optimal policy")
    parser.add_paramsfile_single()
    parser.add_cycle_length()
    parser.add_iterations(default=500)
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
        targets_mean, targets_dec = (dict() for i in range(2))
        emit_dec = calc_emit_dec(mdp_fh, y_emit, y)

        years_sampled = [i*y for i in range(0, mdp_fh.n_years//y)]
        # Sampled from mean of optimal policy.
        targets_mean['x'] = years_sampled
        targets_mean['y'] = [y_emit[i] for i in years_sampled]
        # Decrement evenly to align with optimal policy.
        targets_dec['x'] = years_sampled
        targets_dec['y'] = [y_emit[0] - i*emit_dec for i in range(mdp_fh.n_years//y)]

        targets_dir = Path("visuals/v{}/targets".format(DIR_VERSION))
        name = args.paramsfile.replace("_exp", "").replace("_lin", "")

        tf_mean = targets_dir / "e_v{}_{}_{}_mean.txt".format(DIR_VERSION, y, name)
        with open(tf_mean, 'w+') as targetsfile:
            targetsfile.write(str(targets_mean))

        tf_dec = targets_dir / "e_v{}_{}_{}_dec.txt".format(DIR_VERSION, y, name)
        with open(tf_dec, 'w+') as targetsfile:
            targetsfile.write(str(targets_dec))
        targetsfile.close()


if __name__ == "__main__":
    main(sys.argv[1:])
