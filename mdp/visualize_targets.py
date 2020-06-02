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


def main(argv):
    parser = MdpArgs(description="plot CO2 and RES actual vs. target of following MDP instance optimal policy")
    parser.add_model_version()
    parser.add_paramfile_multiple()
    parser.add_use_data()
    parser.add_emissions_target()
    parser.add_time_range()
    parser.add_iterations()
    parser.add_confidence_interval()
    parser.add_save()
    args = parser.get_args()

    if not parser.check_version(4, 4, "error: visualize_co2 only supported for MDP V4 or higher."):
        sys.exit(1)

    if not parser.check_paramfile_multiple():
        sys.exit(2)

    if not args.targetsfile:
        print("error: must pass in targetsfile.")
        sys.exit(4)

    params_all = cl.get_params_multiple(args.version, args.paramsfiles)
    mdp_model = cl.get_mdp_model(args.version, params_all)

    t_range = cl.get_time_range(args, params_all[0])
    if not t_range:
        sys.exit(3)

    np.set_printoptions(linewidth=300)
    visuals_dir = Path("visuals/v{}/plots".format(args.version))

    mdp_data = MdpDataGatherer(mdp_model, args.iterations, t_range, ci_type="STD")

    if args.usedata:
        y_res, y_emit = ([] for i in range(2))
        for paramfile in args.paramsfiles:
            data = cl.get_mdp_data(args.version, paramfile)
            y_res.append(data['res_penetration'])
            y_emit.append(data['co2_emissions'])
    else:
        mdp_fh_all = cl.get_mdp_instance_multiple(mdp_model, params_all)
        y_res = [mdp_data.res_penetration(mdp_fh) for mdp_fh in mdp_fh_all]
        y_emit = [mdp_data.co2_emissions(mdp_fh) for mdp_fh in mdp_fh_all]

    targets = cl.get_emissions_target(args.version, args.targetsfile)
    y_target = cl.adjust_emissions_target_timeline(targets, params_all[0]['co2_tax_cycle'], t_range)

    x = mdp_data.get_time_range(t_range)

    mdp_plot = MdpPlotter()
    # RES penetration
    mdp_plot.initialize("RES Penetration", "Time (years)", "RES Penetration (%)")
    mdp_plot.plot_lines(x, y_res, args.paramsfiles, CI=args.CI)
    fig_res = mdp_plot.finalize()
    # CO2 emissions (actual vs. target)
    mdp_plot.initialize("Annual CO2 Emissions", "Time (years)", "CO2 Emissions (ton/yr)")
    mdp_plot.plot_lines(x, y_emit, args.paramsfiles, CI=args.CI)
    mdp_plot.add_fixed_line(x, y_target, "Target")
    fig_emit = mdp_plot.finalize()

    if args.save:
        fig_res.savefig(visuals_dir / "g_v{}_res_{}.png".format(args.version))
        fig_emit.savefig(visuals_dir / "g_v{}_target_{}.png".format(args.version))
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
