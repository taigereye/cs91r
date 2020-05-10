import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import mdp.visuals.mdp_visualize as mv
from mdp.models.mdp_v2 import MdpModelV2
from mdp.models.mdp_v3 import MdpModelV3


def format_param_names(param_file):
    pf = param_file
    pf = pf.replace("co2_tax_", "")
    if pf.islower():
        if pf == "stern":
            return "Stern Review"
        else:
            return pf.title()
    else:
        return pf


def main(argv):
    parser = argparse.ArgumentParser(description="plot costs of following MDP instance optimal policy")
    parser.add_argument("-m", "--version", help="MDP model version", type=int)
    parser.add_argument("-p", "--paramsfiles", help="list of txt file with version specific params as dict", nargs='*', action='store')
    parser.add_argument("-t", "--timerange", help="see specific time range", nargs=2, type=int, default=None)
    parser.add_argument("-i", "--iterations", help="number of simulations of tech stage transition", type=int, default=200)
    parser.add_argument("-e", "--CO2", help="limit on annual CO2 emissions", type=int, default=0)
    parser.add_argument("--save", help="save plots as png files", action='store_true')
    args = parser.parse_args()

    if int(args.version) < 2:
        print("error: plot_storage_sensitivity only supported for MDP v2 or higher.")
        sys.exit(1)

    params_co2_taxes = []
    for paramsfile in args.paramsfiles:
        params_dir = Path("results/v{}/params".format(args.version))
        pf = params_dir / "p_v{}_{}.txt".format(args.version, paramsfile)
        with open(pf, 'r') as paramsfile:
            params = eval(paramsfile.read())
        paramsfile.close()
        params_co2_taxes.append(params)
    # param_names = [format_param_names(pf) for pf in args.paramsfiles]
    param_names = ["Flat", "Exponential: 1%/yr", "Exponential: 5%/yr", "Linear: 10 USD/yr", "Linear: 20 USD/yr", "Exponential: 10%/yr"]

    mdp_model = None
    if int(args.version) == 2:
        mdp_model = MdpModelV2()
        p_adv_vary = False
    if int(args.version) == 3:
        mdp_model = MdpModelV3()
        p_adv_vary = True

    assert(mdp_model is not None)
    assert(mdp_model.param_names == list(params.keys()))

    mdp_fh_co2_taxes = []
    for params in params_co2_taxes:
        mdp_fh = mdp_model.run_fh(params)
        mdp_fh_co2_taxes.append(mdp_fh)

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

    fig_co2_emit = mv.opt_policy_co2_emit(mdp_fh_co2_taxes, [t0, tN], args.iterations, param_names, CO2=args.CO2, p_adv_vary=p_adv_vary)

    if args.save:
        fig_co2_emit.savefig(visuals_dir / "g_v{}_compare_co2_emit_ann{}.png".format(args.version, paramsfile))
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
