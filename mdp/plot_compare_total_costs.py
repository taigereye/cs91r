import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import mdp.visuals.mdp_visualize as mv
from mdp.models.mdp_v2 import MdpModelV2
from mdp.models.mdp_v3 import MdpModelV3


def main(argv):
    parser = argparse.ArgumentParser(description="plot costs of following MDP instance optimal policy")
    parser.add_argument("-m", "--version", help="MDP model version", type=int)
    parser.add_argument("-p", "--paramsfiles", help="list of txt files with version specific params as dict", nargs='*', action='store')
    parser.add_argument("-t", "--timerange", help="see specific time range", nargs=2, type=int, default=None)
    parser.add_argument("-i", "--iterations", help="number of simulations of tech stage transition", type=int, default=200)
    parser.add_argument("--save", help="save plots as png files", action='store_true')
    args = parser.parse_args()

    if int(args.version) < 2:
        print("error: plot_compare_total_costs only supported for MDP v2 or higher.")
        sys.exit(1)

    if len(args.paramsfiles) == 0:
        print("error: plot_compare_total_costs requires at least one parameters file.")
        sys.exit(2)

    params_costs = []
    for paramsfile in args.paramsfiles:
        params_dir = Path("results/v{}/params".format(args.version))
        pf = params_dir / "p_v{}_{}.txt".format(args.version, paramsfile)
        with open(pf, 'r') as paramsfile:
            params = eval(paramsfile.read())
        paramsfile.close()
        params_costs.append(params)
    param_names = [mv.format_param_names(pf) for pf in args.paramsfiles]

    param_type = "_".join(args.paramsfiles)

    mdp_model = None
    if int(args.version) == 2:
        mdp_model = MdpModelV2()
    if int(args.version) == 3:
        mdp_model = MdpModelV3()

    assert(mdp_model is not None)
    assert(mdp_model.param_names == list(params.keys()))

    mdp_fh_costs = []
    for params in params_costs:
        mdp_fh = mdp_model.run_fh(params)
        mdp_fh_costs.append(mdp_fh)

    if args.timerange:
        t0, tN = args.timerange
        t0 = max(0, t0-1)
        if tN - t0 > mdp_fh.n_years:
            print("error: time range {}-{} out of range: {}".format(t0, tN, mdp_fh.n_tech_stages))
            sys.exit(2)
    else:
        t0 = 0
        tN = mdp_fh_costs[0].n_years

    np.set_printoptions(linewidth=300)
    visuals_dir = Path("visuals/v{}/plots".format(args.version))

    fig_ann = mv.total_cost_ann_cumu(mdp_fh_costs, param_names, [t0, tN], args.iterations, is_annual=True, p_adv_vary=True)
    fig_cum = mv.total_cost_ann_cumu(mdp_fh_costs, param_names, [t0, tN], args.iterations, is_annual=False, p_adv_vary=True)
    fig_both = mv.total_cost_combine(mdp_fh_costs, param_names, [t0, tN], args.iterations, p_adv_vary=True)

    if args.save:
        fig_ann.savefig(visuals_dir / "g_v{}_cost_ann_{}.png".format(args.version, param_type))
        fig_cum.savefig(visuals_dir / "g_v{}_cost_cumu_{}.png".format(args.version, param_type))
        fig_both.savefig(visuals_dir / "g_v{}_cost_combine_{}.png".format(args.version, param_type))
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
