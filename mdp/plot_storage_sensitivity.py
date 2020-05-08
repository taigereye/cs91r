import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import mdp.visuals.mdp_visualize as mv
from mdp.models.mdp_v2 import MdpModelV2


def main(argv):
    parser = argparse.ArgumentParser(description="plot costs of following MDP instance optimal policy")
    parser.add_argument("-m", "--version", help="MDP model version", type=int)
    parser.add_argument("-p", "--paramsfile", help="txt file with version specific params as dict")
    parser.add_argument("-t", "--timerange", help="see specific time range", nargs=2, type=int, default=None)
    parser.add_argument("-s", "--reductions", help="fractional reductions in storage capital costs", nargs='+', type=float)
    parser.add_argument("-b", "--budget", help="annual budget for renewable plants", type=int, default=None)
    parser.add_argument("-r", "--RESpenetration", help="target renewable penetration", type=int, default=None)
    parser.add_argument("--save", help="save plots as png files", action='store_true')
    args = parser.parse_args()

    if int(args.version) < 2:
        print("error: plot_storage_sensitivity only supported for MDP v2 or higher.")
        sys.exit(1)

    params_dir = Path("results/v{}/params".format(args.version))
    pf = params_dir / "p_v{}_{}.txt".format(args.version, args.paramsfile)
    with open(pf, 'r') as paramsfile:
        params = eval(paramsfile.read())
    paramsfile.close()

    mdp_model = None
    if int(args.version) == 2:
        mdp_model = MdpModelV2()

    assert(mdp_model is not None)
    assert(mdp_model.param_names == list(params.keys()))

    mdp_fh_reduced = []
    for frac in args.reductions:
        params_reduced = mv.reduce_storage_costs_params(params, frac)
        mdp_fh = mdp_model.run_fh(params_reduced)
        mdp_fh_reduced.append(mdp_fh)

    if args.timerange:
        t0, tN = args.timerange
        t0 = max(0, t0-1)
        if tN - t0 > mdp_fh.n_years:
            print("error: time range {}-{} out of range: {}".format(t0, tN, mdp_fh.n_tech_stages))
            sys.exit(3)
    else:
        t0 = 0
        tN = mdp_fh.n_years

    np.set_printoptions(linewidth=300)
    visuals_dir = Path("visuals/v{}/plots".format(args.version))

    (fig_budget_annual, fig_budget_cum), fig_res = mv.storage_reductions_wrapper(mdp_fh_reduced, [t0, tN], args.reductions, budget=args.budget, RES=args.RESpenetration)

    if args.save:
        fig_budget_annual.savefig(visuals_dir / "g_v{}_storage_reductions_budget_ann{}.png".format(args.version, paramsfile))
        fig_budget_cum.savefig(visuals_dir / "g_v{}_storage_reductions_budget_cum{}.png".format(args.version, paramsfile))
        fig_res.savefig(visuals_dir / "g_v{}_storage_reductions_RESpenetration_{}.png".format(args.version, paramsfile))
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
