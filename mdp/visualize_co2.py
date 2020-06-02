import sys

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import mdp.analysis.MdpCLI as cl
from mdp.analysis.MdpCLI import MdpArgs
from mdp.visuals.MdpViz import MdpDataGatherer, MdpPlotter


def main(argv):
    parser = MdpArgs(description="plot CO2 related metrics of following MDP instance optimal policy")
    parser.add_model_version()
    parser.add_paramfile_multiple()
    parser.add_use_data()
    parser.add_time_range()
    parser.add_iterations()
    parser.add_confidence_interval()
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

    np.set_printoptions(linewidth=300)
    visuals_dir = Path("visuals/v{}/plots".format(args.version))

    mdp_data = MdpDataGatherer(mdp_model, args.iterations, t_range, ci_type="QRT")

    if args.usedata:
        y_l, y_price, y_tax, y_emit = ([] for i in range(4))
        for paramfile in args.paramsfiles:
            data = cl.get_mdp_data(args.version, paramfile)
            y_l.append(data['tax_level']['mean'])
            y_price.append(data['co2_price'])
            y_tax.append(data['co2_tax'])
            y_emit.append(data['co2_emissions'])
    else:
        mdp_fh_all = cl.get_mdp_instance_multiple(mdp_model, params_all)
        y_l = [mdp_data.get_state_variable(mdp_fh, 'l')['mean'] for mdp_fh in mdp_fh_all]
        y_price = [mdp_data.co2_current_price(mdp_fh) for mdp_fh in mdp_fh_all]
        y_tax = [mdp_data.co2_tax_collected(mdp_fh) for mdp_fh in mdp_fh_all]
        y_emit = [mdp_data.co2_emissions(mdp_fh) for mdp_fh in mdp_fh_all]

    x = mdp_data.get_time_range(t_range)

    figs_co2 = []
    mdp_plot = MdpPlotter()
    # Tax level
    mdp_plot.initialize("State Variable: Tax Level", "Time (years)", "Tax Delta (USD)")
    mdp_plot.plot_scatter(x, y_l, args.paramsfiles,
                          y_min=-15*(params_all[0]['n_tax_levels']//2), y_max=15*(params_all[0]['n_tax_levels']//2))
    fig = mdp_plot.finalize()
    figs_co2.append(fig)
    # Current CO2 price
    mdp_plot.initialize("Current CO2 Price", "Time (years)", "Cost (USD/ton)")
    mdp_plot.plot_lines(x, y_price, args.paramsfiles, CI=args.CI)
    fig = mdp_plot.finalize()
    figs_co2.append(fig)
    # CO2 tax collected
    mdp_plot.initialize("Annual CO2 Tax Collected", "Time (years)", "Cost (USD/yr)")
    mdp_plot.plot_lines(x, y_tax, args.paramsfiles, CI=args.CI)
    fig = mdp_plot.finalize()
    figs_co2.append(fig)
    # CO2 emissions
    mdp_plot.initialize("Annual CO2 Emissions", "Time (years)", "Cost (ton/yr)")
    mdp_plot.plot_lines(x, y_emit, args.paramsfiles, CI=args.CI)
    fig = mdp_plot.finalize()
    figs_co2.append(fig)

    names = ["level", "price", "tax", "emit"]
    if args.save:
        for fig, name in zip(figs_co2, names):
            fig.savefig(visuals_dir / "g_v{}_co2_{}_{}.png".format(args.version, name, args.paramsfile))
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
