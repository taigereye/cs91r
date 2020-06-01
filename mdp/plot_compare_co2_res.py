import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import mdp.analysis.mdp_visualize as mv
from mdp.models.MdpV2 import MdpModelV2
from mdp.models.MdpV3 import MdpModelV3
from mdp.models.MdpV4 import MdpModelV4


def main(argv):
    parser = argparse.ArgumentParser(description="plot costs of following MDP instance optimal policy")
    parser.add_argument("-m", "--version", help="MDP model version", type=int)
    parser.add_argument("-p", "--paramsfiles", help="list of txt files with version specific params as dict", nargs='*', action='store')
    parser.add_argument("-t", "--timerange", help="see specific time range", nargs=2, type=int, default=None)
    parser.add_argument("-i", "--iterations", help="number of simulations of tech stage transition", type=int, default=200)
    parser.add_argument("-e", "--CO2", help="limit on annual CO2 emissions", type=int, default=0)
    parser.add_argument("-r", "--RES", help="target RES penetration", type=int, default=0)
    parser.add_argument("--save", help="save plots as png files", action='store_true')
    args = parser.parse_args()

    if int(args.version) < 2:
        print("error: plot_compare_co2_res only supported for MDP V2 or higher.")
        sys.exit(1)

    if len(args.paramsfiles) == 0:
        print("error: plot_compare_co2_res requires at least one parameters file.")
        sys.exit(2)

    params_co2_res = []
    for paramsfile in args.paramsfiles:
        params_dir = Path("results/v{}/params".format(args.version))
        pf = params_dir / "p_v{}_{}.txt".format(args.version, paramsfile)
        with open(pf, 'r') as paramsfile:
            params = eval(paramsfile.read())
        paramsfile.close()
        params_co2_res.append(params)
    param_names = [mv.format_param_names(pf) for pf in args.paramsfiles]

    mdp_model = None
    if int(args.version) == 2:
        mdp_model = MdpModelV2()
        p_adv_vary = False
    elif int(args.version) == 3:
        mdp_model = MdpModelV3()
        p_adv_vary = True
    elif int(args.version) == 4:
        mdp_model = MdpModelV4()
        p_adv_vary = True

    assert(mdp_model is not None)
    assert(mdp_model.param_names == list(params.keys()))

    mdp_fh_co2_res = []
    for params in params_co2_res:
        mdp_fh = mdp_model.run_fh(params)
        mdp_fh_co2_res.append(mdp_fh)

    if args.timerange:
        t0, tN = args.timerange
        t0 = max(0, t0-1)
        if tN - t0 > mdp_fh.n_years:
            print("error: time range {}-{} out of range: {}".format(t0, tN, mdp_fh.n_tech_stages))
            sys.exit(3)
    else:
        t0 = 0
        tN = mdp_fh_co2_res[0].n_years

    np.set_printoptions(linewidth=300)
    visuals_dir = Path("visuals/v{}/plots".format(args.version))

    fig_co2_emit = mv.opt_policy_co2_emit(mdp_fh_co2_res, [t0, tN], args.iterations, param_names, CO2=args.CO2, p_adv_vary=p_adv_vary)
    fig_res = mv.opt_policy_res_percent(mdp_fh_co2_res, [t0, tN], args.iterations, param_names, RES=args.RES, p_adv_vary=p_adv_vary)

    if args.save:
        fig_co2_emit.savefig(visuals_dir / "g_v{}_compare_co2_emit_ann{}.png".format(args.version, paramsfile))
        fig_res.savefig(visuals_dir / "g_v{}_compare_res_ann{}.png".format(args.version, paramsfile))
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
