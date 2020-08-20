import sys

from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import mdp.analysis.MdpCLI as cl
import mdp.analysis.MdpData as mv
from mdp.analysis.MdpCLI import MdpArgs
from mdp.analysis.MdpViz import MdpDataGatherer, MdpPlotter


COMP_VERSION = 3
START_YEAR = 2020


def main(argv):
    parser = MdpArgs(description="plot tax adjusted vs. non-adjusted targets of following optimal MDP policy")
    parser.add_model_version()
    parser.add_paramsfile_multiple()
    parser.add_targetsfile()
    parser.add_time_range()
    parser.add_iterations()
    parser.add_confidence_interval()
    parser.add_use_data()
    parser.add_save()
    args = parser.get_args()

    if not parser.check_version(4, 4, "visualize_adjust only supported for MDP V3 with MDP V4 or higher."):
        sys.exit(1)

    if not parser.check_paramfile_multiple():
        sys.exit(2)

    # MDP V3.
    params3 = cl.get_params_single(COMP_VERSION, args.paramsfiles[0])
    mdp_model3 = cl.get_mdp_model(COMP_VERSION, [params3])
    mdp_fh3 = cl.get_mdp_instance_single(mdp_model3, params3)
    # MDP V4.
    params4_all = cl.get_params_multiple(args.version, args.paramsfiles[1:])
    mdp_model4 = cl.get_mdp_model(args.version, params4_all)

    t_range = cl.get_time_range(args, params4_all[0])
    if not t_range:
        sys.exit(3)

    mdp_data = MdpDataGatherer(mdp_model4, args.iterations, t_range)
    if args.confidenceinterval:
        mdp_data.set_ci(ci_type=args.confidenceinterval)

    x = mdp_data.get_time_range(t_range)

    _, y_res3, y_emit3, y_tax3 = mv.avg_co2_probabilistic_v(mdp_fh3, t_range[0], t_range[1],
                                                            args.iterations, True, res_percent=True)
    y_res3 = mdp_data.convert_to_percent(mdp_data.calc_data_bounds(y_res3))
    y_emit3 = mdp_data.calc_data_bounds(y_emit3)
    y_tax3 = mdp_data.calc_data_bounds(y_tax3)

    y_res4, y_emit4, y_tax4 = ([] for i in range(3))
    if args.usedata:
        for pf in args.paramsfiles[1:]:
            data = cl.get_mdp_data(args.version, pf)
            y_res4.append(mdp_data.convert_to_percent(mdp_data.get_data_component(data, 'res_penetration')))
            y_emit4.append(mdp_data.get_data_component(data, 'co2_emissions'))
            y_tax4.append(mdp_data.get_data_component(data, 'co2_tax'))
    else:
        mdp_fh_all = cl.get_mdp_instance_multiple(mdp_model4, params4_all)
        for mdp_fh4 in mdp_fh_all:
            y_res4.append(mdp_data.convert_to_percent(mdp_data.calc_data_bounds(mdp_data.res_penetration(mdp_fh4))))
            y_emit4.append(mdp_data.calc_data_bounds(mdp_data.co2_emissions(mdp_fh4)))
            y_tax4.append(mdp_data.calc_data_bounds(mdp_data.co2_tax_collected(mdp_fh4)))

    y_res = [y_res3] + y_res4
    y_emit = [y_emit3] + y_emit4
    y_tax = [y_tax3] + y_tax4

    if args.targetsfile:
        targets = cl.get_emissions_target(args.version, args.targetsfile)
        targets['x'] = [x + START_YEAR for x in targets['x']]
        targets['x'] = targets['x'][0:7]
        targets['y'] = targets['y'][0:7]
    else:
        targets = None

    # params_names = args.paramsfiles
    params_names = ["{}: static".format(args.paramsfiles[0])] + ["{}: adjusted".format(pf) for pf in args.paramsfiles[1:]]
    colors = None

    figs_compare = []
    mdp_plot = MdpPlotter()
    # RES penetration
    mdp_plot.initialize("RES Penetration", "Time (years)", "RES Penetration (%)")
    mdp_plot.plot_lines(x, y_res, params_names, y_max=100, colors=colors, CI=args.confidenceinterval)
    fig = mdp_plot.finalize()
    figs_compare.append(fig)
    # CO2 emissions
    mdp_plot.initialize("Annual CO2 Emissions", "Time (years)", "CO2 (million ton/yr)")
    mdp_plot.plot_lines(x, y_emit, params_names, scale=1e6, colors=colors, CI=args.confidenceinterval)
    if targets:
        mdp_plot.add_scatter_points(targets['x'], targets['y'], "Emissions target", scale=1e6, marker='o')
    fig = mdp_plot.finalize()
    figs_compare.append(fig)
    # CO2 tax collected
    mdp_plot.initialize("Annual CO2 Tax Collected", "Time (years)", "Cost (billion USD/yr)")
    mdp_plot.plot_lines(x, y_tax, params_names, scale=1e9, colors=colors, CI=args.confidenceinterval)
    fig = mdp_plot.finalize()
    figs_compare.append(fig)

    names = ['res', 'emit', 'target']
    visuals_dir = Path("visuals/v{}/plots".format(args.version))
    if args.save:
        for fig, name in zip(figs_compare, names):
            fig.savefig(visuals_dir / "g_v{}_compare_{}.png".format(args.version, name))
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
