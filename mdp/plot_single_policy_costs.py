import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import mdp.visuals.mdp_visualize as mv
from mdp.models.mdp_v2 import MdpModelV2
from mdp.models.mdp_v3 import MdpModelV3


COMPONENTS = ["co2_tax",
              "ff_total",
              "res_total",
              "bss_total",
              "phs_total"]

COMPONENTS_GRANULAR = ["co2_tax",
                       "ff_replace",
                       "ff_om",
                       "res_cap",
                       "res_replace",
                       "bss_cap",
                       "bss_om",
                       "phs_cap",
                       "phs_om"]


def main(argv):
    parser = argparse.ArgumentParser(description="plot costs of following MDP instance optimal policy")
    parser.add_argument("-m", "--version", help="MDP model version", type=int)
    parser.add_argument("-p", "--paramsfile", help="txt file with version specific params as dict")
    parser.add_argument("-t", "--timerange", help="see specific time range", nargs=2, type=int, default=None)
    parser.add_argument("-v", "--techstage", help="see single tech stage", type=int, default=None)
    parser.add_argument("-a", "--policy", help="txt file with policy as list", default=None)
    parser.add_argument("--granular", help="more granular cost component breakdown", action='store_true')
    parser.add_argument("--save", help="save plots as png files", action='store_true')
    args = parser.parse_args()

    if int(args.version) < 2:
        print("error: plot_single_policy_costs only supported for MDP v2 or higher.")
        sys.exit(1)

    params_dir = Path("results/v{}/params".format(args.version))
    pf = params_dir / "p_v{}_{}.txt".format(args.version, args.paramsfile)
    with open(pf, 'r') as paramsfile:
        params = eval(paramsfile.read())
    paramsfile.close()

    mdp_model = None
    if int(args.version) == 2:
        mdp_model = MdpModelV2()
    if int(args.version) == 3:
        mdp_model = MdpModelV3()

    assert(mdp_model is not None)
    assert(mdp_model.param_names == list(params.keys()))
    mdp_fh = mdp_model.run_fh(params)

    if args.techstage:
        if args.techstage < 0 or args.techstage >= mdp_fh.n_tech_stages:
            print("error: tech stage {} out of range: {}".format(args.techstage, mdp_fh.n_tech_stages))
            sys.exit(2)

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
    else:
        policy_type = "optimal"

    if args.techstage is not None:
        if args.policy:
            policy = mv.get_arb_policy_trajectory(arb_policy, args.techstage)
        else:
            policy = mv.get_opt_policy_trajectory(mdp_fh, args.techstage)
        v_str = str(args.techstage)
    else:
        policy = []
        for v in np.arange(mdp_fh.n_tech_stages):
            if args.policy:
                policy.append(mv.get_arb_policy_trajectory(arb_policy, v))
            else:
                policy.append(mv.get_opt_policy_trajectory(mdp_fh, v))
        v_str = "all"

    if args.granular:
        components = COMPONENTS_GRANULAR
    else:
        components = COMPONENTS

    np.set_printoptions(linewidth=300)
    visuals_dir = Path("visuals/v{}/plots".format(args.version))

    fig_breakdown = mv.total_cost_wrapper(mdp_fh, policy, policy_type, [t0, tN], v=args.techstage)
    fig_percents = mv.cost_breakdown_wrapper(mdp_fh, policy, policy_type, components, [t0, tN], v=args.techstage)
    fig_total = mv.cost_breakdown_wrapper(mdp_fh, policy, policy_type, components, [t0, tN], v=args.techstage, percent=True)

    if args.save:
        fig_breakdown.savefig(visuals_dir / "g_v{}_{}_breakdown_{}_{}.png".format(args.version, policy_type, paramsfile, v_str))
        fig_percents.savefig(visuals_dir / "g_v{}_{}_percents_{}_{}.png".format(args.version, policy_type, paramsfile, v_str))
        fig_total.savefig(visuals_dir / "g_v{}_{}_total_{}_{}.png".format(args.version, policy_type, paramsfile, v_str))
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
