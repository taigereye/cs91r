import sys

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import mdp.analysis.MdpCLI as cl
from mdp.analysis.MdpCLI import MdpArgs
from mdp.visuals.MdpViz import MdpDataGatherer, MdpPlotter


ERR_MSG = "error: visualize_state only supported for MDP V4 or higher."


def main(argv):
    parser = MdpArgs(description="plot state variables of following MDP instance optimal policy")
    parser.add_model_version()
    parser.add_paramfile_multiple()
    parser.add_time_range()
    parser.add_iterations()
    parser.add_save()
    args = parser.get_args()

    if not parser.check_version(2, 4, ERR_MSG):
        sys.exit(1)

    if not parser.check_paramfile_multiple(ERR_MSG):
        sys.exit(2)

    params_all = cl.get_params_multiple(args.version, args.paramsfiles)
    mdp_model = cl.get_mdp_model(args.version, params_all)
    mdp_fh_all = cl.get_mdp_instance_multiple(mdp_model, params_all)

    t_range = cl.get_time_range(args, mdp_fh_all[0])
    if not t_range:
        sys.exit(3)

    np.set_printoptions(linewidth=300)
    visuals_dir = Path("visuals/v{}/plots".format(args.version))

    mdp_data = MdpDataGatherer(mdp_model, args.iterations, t_range)

    y_state = []
    var_codes = ['v', 'r', 'l', 'e']
    for code in var_codes:
        y_all = [mdp_data.get_state_variable(mdp_fh, code) for mdp_fh in mdp_fh_all]
        y_state.append(y_all)

    x = mdp_data.get_time_range(t_range)

    figs_state = []
    mdp_plot = MdpPlotter()
    # Tech stage
    mdp_plot.initialize("State Variable: Tech Stage", "Time (years)", "Tech Stage")
    mdp_plot.plot_bars(x, y_state[0], args.paramsfiles,
                       y_min=0, y_max=mdp_fh_all[0].n_tech_stages-1)
    fig = mdp_plot.finalize()
    figs_state.append(fig)
    # RES plants
    mdp_plot.initialize("State Variable: RES Plants", "Time (years)", "RES Plants")
    mdp_plot.plot_bars(x, y_state[1], args.paramsfiles,
                       y_min=0, y_max=mdp_fh_all[0].n_plants)
    fig = mdp_plot.finalize()
    figs_state.append(fig)
    # Tax level
    mdp_plot.initialize("State Variable: Tax Level", "Time (years)", "Tax Delta (USD)")
    mdp_plot.plot_scatter(x, y_state[2], args.paramsfiles,
                          y_min=-15*(mdp_fh_all[0].n_tax_levels//2), y_max=15*(mdp_fh_all[0].n_tax_levels//2))
    fig = mdp_plot.finalize()
    figs_state.append(fig)
    # Tax adjustment
    mdp_plot.initialize("State Variable: Tax Adjustment", "Time (years)", "Tax Adjustment")
    mdp_plot.plot_scatter(x, y_state[3], args.paramsfiles,
                          y_min=-1.5*(mdp_fh_all[0].n_tax_levels//2), y_max=1.5*(mdp_fh_all[0].n_tax_levels//2))
    fig = mdp_plot.finalize()
    figs_state.append(fig)

    if args.save:
        for fig, code in zip(figs_state, var_codes):
            fig.savefig(visuals_dir / "g_v{}_state_{}.png".format(args.version, code))
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
