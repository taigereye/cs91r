import sys

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import mdp.analysis.MdpCLI as cl
from mdp.analysis.MdpCLI import MdpArgs
from mdp.visuals.MdpViz import MdpDataGatherer, MdpPlotter


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

ERR_MSG = "visualize_costs supported for MDP V4 only."


def main(argv):
    parser = MdpArgs(description="plot costs of following MDP instance optimal policy")
    parser.add_model_version()
    parser.add_paramfile_multiple()
    parser.add_time_range()
    parser.add_iterations()
    parser.add_confidence_interval()
    parser.add_granular()
    parser.add_save()
    args = parser.get_args()

    if not parser.check_version(4, 4, ERR_MSG):
        sys.exit(1)

    if not parser.check_paramfile_multiple(ERR_MSG):
        sys.exit(2)

    params_all = cl.get_params_multiple(args.version, args.paramsfiles)
    mdp_model = cl.get_mdp_model(args.version, params_all)
    mdp_fh_all = cl.get_mdp_instance_multiple(mdp_model, params_all)

    t_range = cl.get_time_range(args, mdp_fh_all[0])
    if not t_range:
        sys.exit(3)

    if args.granular:
        components = COMPONENTS_GRANULAR
    else:
        components = COMPONENTS

    np.set_printoptions(linewidth=300)
    visuals_dir = Path("visuals/v{}/plots".format(args.version))

    mdp_data = MdpDataGatherer(mdp_model, args.iterations, t_range, ci_type="QRT")

    y_total = [mdp_data.cost_total(mdp_fh) for mdp_fh in mdp_fh_all]
    y_breakdown = [mdp_data.cost_breakdown_components(mdp_fh, components) for mdp_fh in mdp_fh_all]
    y_percents = [mdp_data.cost_breakdown_components(mdp_fh, components, is_percent=True) for mdp_fh in mdp_fh_all]

    x = mdp_data.get_time_range(t_range)

    mdp_plot = MdpPlotter()
    # Total cost
    mdp_plot.initialize("Total Annual Cost", "Time (years)", "Cost (USD/yr)")
    mdp_plot.plot_lines(x, y_total, args.paramsfiles, CI=args.CI)
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
        fig_total.savefig(visuals_dir / "g_v{}_total.png".format(args.version))
        fig_breakdown.savefig(visuals_dir / "g_v{}_breakdown.png".format(args.version))
        fig_percents.savefig(visuals_dir / "g_v{}_percents.png".format(args.version))
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
