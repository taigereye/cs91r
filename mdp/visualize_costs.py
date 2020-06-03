import sys

from collections import OrderedDict
import matplotlib.pyplot as plt
from pathlib import Path

import mdp.analysis.MdpCLI as cl
from mdp.analysis.MdpCLI import MdpArgs
from mdp.visuals.MdpViz import MdpDataGatherer, MdpPlotter


def main(argv):
    parser = MdpArgs(description="plot costs of following MDP instance optimal policy")
    parser.add_model_version()
    parser.add_paramsfile_multiple()
    parser.add_time_range()
    parser.add_iterations()
    parser.add_use_data()
    parser.add_confidence_interval()
    parser.add_granular()
    parser.add_save()
    args = parser.get_args()

    if not parser.check_version(4, 4, "visualize_costs supported for MDP V4 only."):
        sys.exit(1)

    if not parser.check_paramfile_multiple():
        sys.exit(2)

    params_all = cl.get_params_multiple(args.version, args.paramsfiles)
    mdp_model = cl.get_mdp_model(args.version, params_all)

    t_range = cl.get_time_range(args, params_all[0])
    if not t_range:
        sys.exit(3)

    if args.granular:
        components = cl.COMPONENTS_GRANULAR
    else:
        components = cl.COMPONENTS

    mdp_data = MdpDataGatherer(mdp_model, args.iterations, t_range)
    if args.confidenceinterval:
        mdp_data.set_ci(ci_type=args.confidenceinterval)

    x = mdp_data.get_time_range(t_range)

    y_total, y_breakdown, y_percents = ([] for i in range(3))
    if args.usedata:
        for paramfile in args.paramsfiles:
            data = cl.get_mdp_data(args.version, paramfile)
            y_total.append(mdp_data.get_data_component(data, 'cost_total'))
            y_breakdown.append(mdp_data.get_data_component(data, 'cost_breakdown'))
            y_percents.append(mdp_data.get_data_component(data, 'cost_percent'))
    else:
        mdp_fh_all = cl.get_mdp_instance_multiple(mdp_model, params_all)
        for mdp_fh in mdp_fh_all:
            y_total.append(mdp_data.calc_data_bounds(mdp_data.cost_total(mdp_fh)))
            y_breakdown.append(mdp_data.cost_breakdown_components(mdp_fh, components))
            y_percents.append(mdp_data.cost_breakdown_components(mdp_fh, components, is_percent=True))

    mdp_plot = MdpPlotter()
    # Total cost
    mdp_plot.initialize("Total Annual Cost", "Time (years)", "Cost (USD/yr)")
    mdp_plot.plot_lines(x, y_total, args.paramsfiles, CI=args.confidenceinterval)
    fig_total = mdp_plot.finalize()
    # Absolute cost breakdown
    mdp_plot.initialize("Absolute Cost Breakdown", "Time (years)", "Cost (USD/yr)")
    mdp_plot.plot_stacked_bars(x, y_breakdown, components)
    fig_breakdown = mdp_plot.finalize()
    # Percentage cost breakdown
    mdp_plot.initialize("Percentage Cost Breakdown", "Time (years)", "Cost (%/yr)")
    mdp_plot.plot_stacked_bars(x, y_percents, components)
    fig_percents = mdp_plot.finalize()

    visuals_dir = Path("visuals/v{}/plots".format(args.version))
    if args.save:
        fig_total.savefig(visuals_dir / "g_v{}_total.png".format(args.version))
        fig_breakdown.savefig(visuals_dir / "g_v{}_breakdown.png".format(args.version))
        fig_percents.savefig(visuals_dir / "g_v{}_percents.png".format(args.version))
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
