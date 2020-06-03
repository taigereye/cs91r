import sys

from collections import OrderedDict
from sklearn.metrics import mean_squared_error

import mdp.analysis.MdpCLI as cl
from mdp.analysis.MdpCLI import MdpArgs
from mdp.visuals.MdpViz import MdpDataGatherer


def main(argv):
    parser = MdpArgs(description="find set of params with least MSE delta from target CO2 emissions")
    parser.add_paramsfile_multiple()
    parser.add_model_version()
    parser.add_targetsfile()
    parser.add_iterations()
    parser.add_use_data()
    args = parser.get_args()

    if not parser.check_version(4, 4, "get_mse_emissions only supported for MDP V4 or higher."):
        sys.exit(1)

    if not parser.check_paramfile_multiple():
        sys.exit(2)

    params_all = cl.get_params_multiple(args.version, args.paramsfiles)
    mdp_model = cl.get_mdp_model(args.version, params_all)

    t_range = [0, params_all[0]['n_years']]

    mdp_data = MdpDataGatherer(mdp_model, args.iterations, t_range)

    y_emit = []
    if args.usedata:
        for paramfile in args.paramsfiles:
            data = cl.get_mdp_data(args.version, paramfile)
            y_emit.append(mdp_data.get_data_component(data, 'co2_emissions', mean_only=True))
    else:
        mdp_fh_all = cl.get_mdp_instance_multiple(mdp_model, params_all)
        for mdp_fh in mdp_fh_all:
            y_emit.append(cl.calc_data_bounds(mdp_data.co2_emissions(mdp_fh))['mean'])

    targets = cl.get_emissions_target(args.version, args.targetsfile)
    y_target = cl.adjust_emissions_target_timeline(targets, params_all[0]['co2_tax_cycle'], t_range)

    y_mse = [mean_squared_error(y_target, y, squared=True) for y in y_emit]
    y_rmse = [mean_squared_error(y_target, y, squared=False) for y in y_emit]

    print("\nCO2 emissions MSE and RMSE:\n")
    for mse, rmse, paramsfile in zip(y_mse, y_rmse, args.paramsfiles):
        print("{}: {:.3e} {:.3e}".format(paramsfile, mse, rmse))

    print("\n\nParams closest to target emissions:\n")
    idx = y_mse.index(min(y_mse))
    print("{}: {:.3e} {:.3e}".format(args.paramsfiles[idx], y_mse[idx], y_rmse[idx]))


if __name__ == "__main__":
    main(sys.argv[1:])
