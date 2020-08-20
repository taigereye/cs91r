import sys

from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import mdp.analysis.MdpCLI as cl
from mdp.analysis.MdpCLI import MdpArgs
from mdp.analysis.MdpViz import MdpDataGatherer, MdpPlotter


START_YEAR = 2020


def main(argv):
    parser = MdpArgs(description="plot CO2 and RES actual vs. target of following MDP instance optimal policy")
    parser.add_model_version()
    parser.add_paramsfile_multiple()
    parser.add_targetsfile()
    parser.add_time_range()
    parser.add_iterations()
    parser.add_confidence_interval()
    parser.add_use_data()
    parser.add_save()
    args = parser.get_args()

    if not parser.check_version(4, 4, "error: visualize_co2 only supported for MDP V4 or higher."):
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

    y_res, y_emit = ([] for i in range(2))
    if args.usedata:
        for pf in args.paramsfiles:
            data = cl.get_mdp_data(args.version, pf)
            y_res.append(mdp_data.convert_to_percent(mdp_data.get_data_component(data, 'res_penetration')))
            y_emit.append(mdp_data.get_data_component(data, 'co2_emissions'))
    else:
        mdp_fh_all = cl.get_mdp_instance_multiple(mdp_model, params_all)
        for mdp_fh in mdp_fh_all:
            y_res.append(mdp_data.convert_to_percent(mdp_data.calc_data_bounds(mdp_data.res_penetration(mdp_fh))))
            y_emit.append(mdp_data.calc_data_bounds(mdp_data.co2_emissions(mdp_fh)))

    if args.targetsfile:
        targets = cl.get_emissions_target(args.version, args.targetsfile)
        targets['x'] = [x + START_YEAR for x in targets['x']]
    else:
        targets = None

    params_names = args.paramsfiles

    mdp_plot = MdpPlotter()
    # RES penetration
    mdp_plot.initialize("RES Penetration", "Time (years)", "RES Penetration (%)")
    mdp_plot.plot_lines(x, y_res, params_names, CI=args.confidenceinterval)
    fig_res = mdp_plot.finalize()
    # CO2 emissions (actual vs. target)
    mdp_plot.initialize("Annual CO2 Emissions", "Time (years)", "CO2 Emissions (ton/yr)")
    mdp_plot.plot_lines(x, y_emit, params_names, CI=args.confidenceinterval)
    if targets:
        mdp_plot.add_scatter_points(targets['x'][t_range[0]:t_range[1]], targets['y'][t_range[0]:t_range[1]], "Emissions target")
    fig_emit = mdp_plot.finalize()

    visuals_dir = Path("visuals/v{}/plots".format(args.version))
    if args.save:
        fig_res.savefig(visuals_dir / "g_v{}_res_{}.png".format(args.version))
        fig_emit.savefig(visuals_dir / "g_v{}_target_{}.png".format(args.version))
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
