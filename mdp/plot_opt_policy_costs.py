import getopt
import sys

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import mdp.visuals.mdp_plot as mdplt
from mdp.models.mdp_v2 import MdpModelV2


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "m:p:v:")
    except getopt.GetoptError:
        print('usage: plot_opt_policy_costs.py -m <modelversion> -p <paramsfile> -v <techstage>')
        sys.exit(1)

    version = str(opts[0][1])
    if int(version) < 1:
        print("error: plot_opt_policy_costs only supported for MDP v2 or higher.")
        sys.exit(2)

    params_dir = Path("results/params_v" + version + "/")
    pf = params_dir / opts[1][1]
    with open(pf, 'r') as paramsfile:
        params = eval(paramsfile.read())
    paramsfile.close()

    mdp_fh = None
    if int(version) == 2:
        mdp_model = MdpModelV2()

    assert(mdp_model is not None)
    assert(mdp_model.param_names == list(params.keys()))
    mdp_fh = mdp_model.run_fh(params)

    v = int(opts[2][1])
    if v < 0 or v >= mdp_fh.n_tech_stages:
        print("error: plot_opt_policy_costs only supported for MDP v2 or higher.")
        sys.exit(2)

    np.set_printoptions(linewidth=300)
    visuals_dir = Path("visuals")
    policy_type = "Optimal Policy"
    pf_name = opts[1][1].split('_')[2].split('.')[0]

    policy = mdplt.get_opt_policy_trajectory(mdp_fh, v)
    fig_breakdown = mdplt.cost_breakdown(mdp_fh, v, policy, policy_type)
    fig_breakdown.savefig(visuals_dir / "opt_policy_{}_cost_breakdown_{}.png".format(pf_name, v))
    fig_percents = mdplt.cost_breakdown(mdp_fh, v, policy, policy_type, percent=True)
    fig_percents.savefig(visuals_dir / "opt_policy_{}_cost_percents_{}.png".format(pf_name, v))
    fig_total = mdplt.total_cost(mdp_fh, v, policy, policy_type)
    fig_total.savefig(visuals_dir / "opt_policy_{}_total_cost_{}.png".format(pf_name, v))
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
