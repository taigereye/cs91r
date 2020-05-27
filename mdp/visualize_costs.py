import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from mdp.visuals.MdpViz import MdpDataGatherer, MdpPlotter
from mdp.models.MdpV4 import MdpModelV4


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
    parser.add_argument("-i", "--iterations", help="number of simulations of tech stage transition", type=int, default=200)
    parser.add_argument("--CI", help="show confidence intervals for single line plots", action='store_true')
    parser.add_argument("--granular", help="more granular cost component breakdown", action='store_true')
    parser.add_argument("--save", help="save plots as png files", action='store_true')
    args = parser.parse_args()

    if int(args.version) < 2:
        print("error: visualize_costs only supported for MDP V4 or higher.")
        sys.exit(1)

    params_dir = Path("results/v{}/params".format(args.version))
    pf = params_dir / "p_v{}_{}.txt".format(args.version, args.paramsfile)
    with open(pf, 'r') as paramsfile:
        params = eval(paramsfile.read())
    paramsfile.close()

    mdp_model = None
    if int(args.version) == 4:
        mdp_model = MdpModelV4()
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
    t_range = [t0, tN]

    if args.granular:
        components = COMPONENTS_GRANULAR
    else:
        components = COMPONENTS

    np.set_printoptions(linewidth=300)
    visuals_dir = Path("visuals/v{}/plots".format(args.version))

    mdp_data = MdpDataGatherer(mdp_model, args.iterations, t_range)
    y_total = mdp_data.cost_total(mdp_fh)
    y_breakdown = mdp_data.cost_breakdown_components(mdp_fh, components)
    y_percents = mdp_data.cost_breakdown_components(mdp_fh, components, is_percent=True)

    x = mdp_data.get_time_range(t_range)

    mdp_plot = MdpPlotter()
    # Total cost
    mdp_plot.initialize("Total Annual Cost: {}".format(args.paramsfile), "Time (years)", "Cost (USD/yr)")
    if args.CI:
        mdp_plot.plot_lines(x, [y_total[0]], [args.paramsfile], y_lower=[y_total[1]], y_upper=[y_total[2]])
    else:
        mdp_plot.plot_lines(x, [y_total[0]], [args.paramsfile])
    fig_total = mdp_plot.finalize()
    # Absolute cost breakdown
    mdp_plot.initialize("Absolute Cost Breakdown: {}".format(args.paramsfile), "Time (years)", "Cost (USD/yr)")
    mdp_plot.plot_stacked_bar(x, y_breakdown, components)
    fig_breakdown = mdp_plot.finalize()
    # Percentage cost breakdown
    mdp_plot.initialize("Percentage Cost Breakdown: {}".format(args.paramsfile), "Time (years)", "Cost (%/yr)")
    mdp_plot.plot_stacked_bar(x, y_percents, components)
    fig_percents = mdp_plot.finalize()

    if args.save:
        fig_total.savefig(visuals_dir / "g_v{}_total_{}.png".format(args.version, args.paramsfile))
        fig_breakdown.savefig(visuals_dir / "g_v{}_breakdown_{}.png".format(args.version, args.paramsfile))
        fig_percents.savefig(visuals_dir / "g_v{}_percents_{}.png".format(args.version, args.paramsfile))
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
