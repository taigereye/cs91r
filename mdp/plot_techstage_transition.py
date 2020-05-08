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
    parser.add_argument("-t", "--timerange", help="see specific time range", nargs=2, type=int, default=None)
    parser.add_argument("-a", "--policy", help="txt file with policy as list", default=None)
    parser.add_argument("-i", "--iterations", help="number of simulations of tech stage transition", type=int, default=200)
    parser.add_argument("--save", help="save plots as png files", action='store_true')
    args = parser.parse_args()

    if int(args.version) < 2:
        print("error: plot_techstage_transition only supported for MDP v2 or higher.")
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
    if int(args.version) == 3:
        mdp_model = MdpModelV3()
        p_adv_vary = True

    assert(mdp_model is not None)
    assert(mdp_model.param_names == list(params.keys()))
    mdp_fh = mdp_model.run_fh(params)

    if args.timerange:
        t0, tN = args.timerange
        t0 = max(0, t0-1)
        if tN - t0 > mdp_fh.n_years:
            print("error: time range {}-{} out of range: {}".format(t0, tN, mdp_fh.n_tech_stages))
            sys.exit(3)
    else:
        t0 = 0
        tN = mdp_fh.n_years

    if args.policy:
        policies_dir = Path("visuals/v{}/policies".format(args.version))
        af = policies_dir / "a_v{}_{}.txt".format(args.version, args.policy)
        with open(af, 'r') as policyfile:
            arb_policy = eval(policyfile.read())
        policyfile.close()
        assert(len(arb_policy) == mdp_fh.n_years)
        policy_type = args.policy
        policy = [mv.get_arb_policy_trajectory(arb_policy, v) for v in np.arange(mdp_fh.n_tech_stages)]
    else:
        policy_type = "optimal"
        policy = [mv.get_opt_policy_trajectory(mdp_fh, v) for v in np.arange(mdp_fh.n_tech_stages)]

    np.set_printoptions(linewidth=300)
    visuals_dir = Path("visuals/v{}/plots".format(args.version))

    fig_fixed_a = mv.policy_plants_all_v(mdp_fh, policy, policy_type, [t0, tN], 'a')
    fig_fixed_r = mv.policy_plants_all_v(mdp_fh, policy, policy_type, [t0, tN], 'r')
    fig_transition = mv.policy_plants_probabilistic_v(mdp_fh, [t0, tN], args.iterations, p_adv_vary=p_adv_vary)

    if args.save:
        fig_fixed_a.savefig(visuals_dir / "g_v{}_{}_fixed_a_{}.png".format(args.version, policy_type, paramsfile))
        fig_fixed_r.savefig(visuals_dir / "g_v{}_{}_fixed_r_{}.png".format(args.version, policy_type, paramsfile))
        fig_transition.savefig(visuals_dir / "g_v{}_transition_{}.png".format(args.version, paramsfile))
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
