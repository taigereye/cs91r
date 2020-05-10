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
    parser.add_argument("-p", "--paramsfile", help="txt file with version specific params as dict")
    parser.add_argument("-s", "--reductions", help="fractional reductions in storage capital costs", nargs='+', type=float)
    parser.add_argument("-t", "--timerange", help="see specific time range", nargs=2, type=int, default=None)
    parser.add_argument("-i", "--iterations", help="number of simulations of tech stage transition", type=int, default=200)
    parser.add_argument("-b", "--budget", help="annual budget for renewable plants", type=int, default=0)
    parser.add_argument("-r", "--RES", help="target renewable penetration", type=int, default=0)
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

    params_storage_reduced = []
    for frac in args.reductions:
        params_reduced = mv.reduce_storage_costs_params(params, frac)
        params_storage_reduced.append(params_reduced)

    mdp_model = None
    if int(args.version) == 2:
        mdp_model = MdpModelV2()
        p_adv_vary = False
    if int(args.version) == 3:
        mdp_model = MdpModelV3()
        p_adv_vary = True

    assert(mdp_model is not None)
    assert(mdp_model.param_names == list(params.keys()))

    mdp_fh_storage_reduced = []
    for params in params_storage_reduced:
        mdp_fh = mdp_model.run_fh(params)
        mdp_fh_storage_reduced.append(mdp_fh)

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

    figs_budget, fig_res = mv.storage_reductions_wrapper(mdp_fh_storage_reduced, [t0, tN], args.iterations, args.reductions, budget=args.budget, RES=args.RES, p_adv_vary=p_adv_vary)

    if args.save:
        figs_budget[0].savefig(visuals_dir / "g_v{}_storage_reductions_budget_ann{}.png".format(args.version, paramsfile))
        figs_budget[1].savefig(visuals_dir / "g_v{}_storage_reductions_budget_cum{}.png".format(args.version, paramsfile))
        fig_res.savefig(visuals_dir / "g_v{}_storage_reductions_RESpenetration_{}.png".format(args.version, paramsfile))
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
