import sys

from collections import OrderedDict
import matplotlib.pyplot as plt
from pathlib import Path

import mdp.analysis.MdpCLI as cl
from mdp.analysis.MdpCLI import MdpArgs
from mdp.analysis.MdpViz import MdpDataGatherer, MdpPlotter


def main(argv):
    parser = MdpArgs(description="plot CO2 related metrics of following MDP instance optimal policy")
    parser.add_model_version()
    parser.add_paramsfile_multiple()
    parser.add_time_range()
    parser.add_iterations()
    parser.add_confidence_interval()
    parser.add_use_data()
    parser.add_save()
    args = parser.get_args()

    if not parser.check_version(4, 4, "visualize_co2 only supported for MDP V4 or higher."):
        sys.exit(1)

    if not parser.check_paramfile_multiple():
        sys.exit(2)

    params_all = cl.get_params_multiple(args.version, args.paramsfiles)
    mdp_model = cl.get_mdp_model(args.version, params_all)

    t_range = cl.get_time_range(args, params_all[0])
    if not t_range:
        sys.exit(3)

    mdp_data = MdpDataGatherer(mdp_model, args.iterations, t_range)
    if args.confidenceinterval:
        mdp_data.set_ci(ci_type=args.confidenceinterval)

    x = mdp_data.get_time_range(t_range)

    y_l, y_price, y_tax, y_emit = ([] for i in range(4))
    if args.usedata:
        for pf in args.paramsfiles:
            data = cl.get_mdp_data(args.version, pf)
            y_l.append(mdp_data.get_data_component(data, 'tax_level', mean_only=True))
            y_price.append(mdp_data.get_data_component(data, 'co2_price'))
            y_tax.append(mdp_data.get_data_component(data, 'co2_tax'))
            y_emit.append(mdp_data.get_data_component(data, 'co2_emissions'))
    else:
        mdp_fh_all = cl.get_mdp_instance_multiple(mdp_model, params_all)
        for mdp_fh in mdp_fh_all:
            y_l.append(mdp_data.calc_data_bounds(mdp_data.get_state_variable(mdp_fh, 'l'))['middle'])
            y_price.append(mdp_data.calc_data_bounds(mdp_data.co2_current_price(mdp_fh)))
            y_tax.append(mdp_data.calc_data_bounds(mdp_data.co2_tax_collected(mdp_fh)))
            y_emit.append(mdp_data.calc_data_bounds(mdp_data.co2_emissions(mdp_fh)))

    params_names = args.paramsfiles
    colors = None

    figs_co2 = []
    mdp_plot = MdpPlotter()
    # Tax level
    mdp_plot.initialize("State Variable: Tax Level", "Time (years)", "Tax Delta (USD)")
    mdp_plot.plot_scatter(x, y_l, params_names, colors=colors)
    fig = mdp_plot.finalize()
    figs_co2.append(fig)
    # Current CO2 price
    mdp_plot.initialize("Current CO2 Price", "Time (years)", "Cost (USD/ton)")
    mdp_plot.plot_lines(x, y_price, params_names, CI=args.confidenceinterval, colors=colors)
    fig = mdp_plot.finalize()
    figs_co2.append(fig)
    # CO2 tax collected
    mdp_plot.initialize("Annual CO2 Tax Collected", "Time (years)", "Cost (USD/yr)")
    mdp_plot.plot_lines(x, y_tax, params_names, CI=args.confidenceinterval, colors=colors)
    fig = mdp_plot.finalize()
    figs_co2.append(fig)
    # CO2 emissions
    mdp_plot.initialize("Annual CO2 Emissions", "Time (years)", "CO2 Emissions (ton/yr)")
    mdp_plot.plot_lines(x, y_emit, params_names, CI=args.confidenceinterval, colors=colors)
    fig = mdp_plot.finalize()
    figs_co2.append(fig)

    names = ["level", "price", "tax", "emit"]
    visuals_dir = Path("visuals/v{}/plots".format(args.version))
    if args.save:
        for fig, name in zip(figs_co2, names):
            fig.savefig(visuals_dir / "g_v{}_co2_{}_{}.png".format(args.version, name, args.paramsfile))
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
