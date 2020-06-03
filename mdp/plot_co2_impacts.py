import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import mdp.analysis.MdpData as mv
from mdp.models.MdpV2 import MdpModelV2
from mdp.models.MdpV3 import MdpModelV3
from mdp.models.MdpV4 import MdpModelV4


def main(argv):
    parser = argparse.ArgumentParser(description="plot costs of following MDP instance optimal policy")
    parser.add_argument("-m", "--version", help="MDP model version", type=int)
    parser.add_argument("-p", "--paramsfile", help="txt file with version specific params as dict")
    parser.add_argument("-t", "--timerange", help="see specific time range", nargs=2, type=int, default=None)
    parser.add_argument("-i", "--iterations", help="number of simulations of tech stage transition", type=int, default=200)
    parser.add_argument("--save", help="save plots as png files", action='store_true')
    args = parser.parse_args()

    if int(args.version) < 2:
        print("error: plot_co2_impacts only supported for MDP V2 or higher.")
        sys.exit(1)

    params_dir = Path("results/v{}/params".format(args.version))
    pf = params_dir / "p_v{}_{}.txt".format(args.version, args.paramsfile)
    with open(pf, 'r') as paramsfile:
        params = eval(paramsfile.read())
    paramsfile.close()

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
    mdp_fh = mdp_model.run_fh(params)

    policy = [mv.get_opt_policy_trajectory(mdp_fh, v) for v in np.arange(mdp_fh.n_tech_stages)]

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

    fig_annual = mv.co2_emit_tax_wrapper(mdp_fh, policy, [t0, tN], args.iterations, is_annual=True, p_adv_vary=p_adv_vary)
    fig_cum = mv.co2_emit_tax_wrapper(mdp_fh, policy, [t0, tN], args.iterations, is_annual=False, p_adv_vary=p_adv_vary)

    if args.save:
        fig_annual.savefig(visuals_dir / "g_v{}_co2_emit_tax_ann_{}.png".format(args.version, paramsfile))
        fig_cum.savefig(visuals_dir / "g_v{}_co2_emit_tax_cum_{}.png".format(args.version, paramsfile))
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
