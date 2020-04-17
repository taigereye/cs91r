import getopt
import sys

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import mdp.visuals.mdp_plot as mdplt
from mdp.models.mdp_v2 import MdpModelV2


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "m:p:a:v:")
    except getopt.GetoptError:
        print('usage: plot_arb_policy_costs.py -m <modelversion> -p <paramsfile> -a <policyfile> -v <techstage>')
        sys.exit(1)

    version = str(opts[0][1])
    if int(version) < 1:
        print("error: plot_arb_policy_costs only supported for MDP v2 or higher.")
        sys.exit(2)

    params_dir = Path("results/v{}/params".format(version))
    pf = params_dir / "p_v{}_{}.txt".format(version, opts[1][1])
    with open(pf, 'r') as paramsfile:
        params = eval(paramsfile.read())
    paramsfile.close()

    mdp_fh = None
    if int(version) == 2:
        mdp_model = MdpModelV2()

    assert(mdp_model is not None)
    assert(mdp_model.param_names == list(params.keys()))
    mdp_fh = mdp_model.run_fh(params)

    v = int(opts[3][1])
    if v < 0 or v >= mdp_fh.n_tech_stages:
        print("error: plot_arb_policy_costs only supported for MDP v2 or higher.")
        sys.exit(2)

    policies_dir = Path("visuals/v{}/policies".format(version))
    af = policies_dir / "a_v{}_{}.txt".format(version, opts[2][1])
    with open(af, 'r') as policyfile:
        policy = eval(policyfile.read())
    policyfile.close()

    assert(len(policy) == mdp_fh.n_years)
    policy = mdplt.add_state_to_policy(policy, v)

    np.set_printoptions(linewidth=300)
    plots_dir = Path("visuals/v{}/plots".format(version))
    policy_type = "Policy: {}".format(opts[2][1])

    fig_breakdown = mdplt.cost_breakdown(mdp_fh, v, policy, policy_type)
    fig_breakdown.savefig(plots_dir / "g_{}_{}_policy_cost_breakdown_{}_{}.png".format(version, opts[2][1], opts[1][1], v))
    fig_percents = mdplt.cost_breakdown(mdp_fh, v, policy, policy_type, percent=True)
    fig_percents.savefig(plots_dir / "g_{}_{}_policy_cost_percents_{}_{}.png".format(version, opts[2][1], opts[1][1], v))
    fig_total = mdplt.total_cost(mdp_fh, v, policy, policy_type)
    fig_total.savefig(plots_dir / "g_{}_{}_policy_total_cost_{}_{}.png".format(version, opts[2][1], opts[1][1], v))
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
