import sys

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import mdp.analysis.MdpCLI as cl
from mdp.analysis.MdpCLI import MdpArgs
from mdp.visuals.MdpViz import MdpDataGatherer, MdpPlotter


def main(argv):
    parser = MdpArgs(description="plot state variables of following MDP instance optimal policy")
    parser.add_model_version()
    parser.add_paramfile_multiple()
    parser.add_use_data()
    parser.add_time_range()
    parser.add_iterations()
    parser.add_save()
    args = parser.get_args()

    if not parser.check_version(4, 4, "visualize_state only supported for MDP V4 or higher."):
        sys.exit(1)

    if not parser.check_paramfile_multiple():
        sys.exit(2)

    params_all = cl.get_params_multiple(args.version, args.paramsfiles)
    mdp_model = cl.get_mdp_model(args.version, params_all)

    t_range = cl.get_time_range(args, params_all[0])
    if not t_range:
        sys.exit(3)

    np.set_printoptions(linewidth=300)
    visuals_dir = Path("visuals/v{}/plots".format(args.version))

    mdp_data = MdpDataGatherer(mdp_model, args.iterations, t_range)

    if args.usedata:
        y_v, y_r, y_l, y_e = ([] for i in range(4))
        for paramfile in args.paramsfiles:
            data = cl.get_mdp_data(args.version, paramfile)
            y_v.append(data['tech_stage']['mean'])
            y_r.append(data['res_plants']['mean'])
            y_l.append(data['tax_level']['mean'])
            y_e.append(data['tax_adjustment']['mean'])
    else:
        mdp_fh_all = cl.get_mdp_instance_multiple(mdp_model, params_all)
        y_v = [mdp_data.get_state_variable(mdp_fh, 'v')['mean'] for mdp_fh in mdp_fh_all]
        y_r = [mdp_data.get_state_variable(mdp_fh, 'r')['mean'] for mdp_fh in mdp_fh_all]
        y_l = [mdp_data.get_state_variable(mdp_fh, 'l')['mean'] for mdp_fh in mdp_fh_all]
        y_e = [mdp_data.get_state_variable(mdp_fh, 'e')['mean'] for mdp_fh in mdp_fh_all]

    x = mdp_data.get_time_range(t_range)

    figs_state = []
    mdp_plot = MdpPlotter()
    # Tech stage
    mdp_plot.initialize("State Variable: Tech Stage", "Time (years)", "Tech Stage")
    mdp_plot.plot_bars(x, y_v, args.paramsfiles,
                       y_min=0, y_max=params_all[0]['n_tech_stages']-1)
    fig = mdp_plot.finalize()
    figs_state.append(fig)
    # RES plants
    mdp_plot.initialize("State Variable: RES Plants", "Time (years)", "RES Plants")
    mdp_plot.plot_bars(x, y_r, args.paramsfiles,
                       y_min=0, y_max=params_all[0]['n_plants'])
    fig = mdp_plot.finalize()
    figs_state.append(fig)
    # Tax level
    mdp_plot.initialize("State Variable: Tax Level", "Time (years)", "Tax Delta (USD)")
    mdp_plot.plot_scatter(x, y_l, args.paramsfiles,
                          y_min=-15*(params_all[0]['n_tax_levels']//2), y_max=15*(params_all[0]['n_tax_levels']//2))
    fig = mdp_plot.finalize()
    figs_state.append(fig)
    # Tax adjustment
    mdp_plot.initialize("State Variable: Tax Adjustment", "Time (years)", "Tax Adjustment")
    mdp_plot.plot_scatter(x, y_e, args.paramsfiles,
                          y_min=-1.5*(params_all[0]['n_tax_levels']//2), y_max=1.5*(params_all[0]['n_tax_levels']//2))
    fig = mdp_plot.finalize()
    figs_state.append(fig)

    codes = ['v', 'r', 'l', 'e']
    if args.save:
        for fig, code in zip(figs_state, codes):
            fig.savefig(visuals_dir / "g_v{}_state_{}.png".format(args.version, code))
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
