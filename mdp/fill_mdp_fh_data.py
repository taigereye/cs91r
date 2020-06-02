import sys

from collections import OrderedDict
from pathlib import Path

import mdp.analysis.MdpCLI as cl
from mdp.analysis.MdpCLI import MdpArgs
from mdp.visuals.MdpViz import MdpDataGatherer


MDP_VERSION = 4
DIR_VERSION = 4

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


def data_dict_array_to_list(data):
    data['mean'] = data['mean'].tolist()
    data['lower'] = data['lower'].tolist()
    data['upper'] = data['upper'].tolist()
    return data


def main(argv):
    parser = MdpArgs(description="run MDP instances and collect stochastic data")
    parser.add_paramfile_single()
    parser.add_iterations()
    parser.add_granular()
    args = parser.get_args()

    if not args.paramsfile:
        print("error: must pass in paramsfile.")
        sys.exit(2)

    params = cl.get_params_single(MDP_VERSION, args.paramsfile)
    mdp_model = cl.get_mdp_model(MDP_VERSION, [params])
    mdp_fh = cl.get_mdp_instance_single(mdp_model, params_all)

    if args.granular:
        components = COMPONENTS_GRANULAR
    else:
        components = COMPONENTS

    t_range = [0, mdp_fh.n_years]
    mdp_data = MdpDataGatherer(mdp_model, args.iterations, t_range, ci_type="QRT")
    # State variables
    y_v = mdp_data.get_state_variable(mdp_fh, 'v')
    y_r = mdp_data.get_state_variable(mdp_fh, 'r')
    y_l = mdp_data.get_state_variable(mdp_fh, 'l')
    y_e = mdp_data.get_state_variable(mdp_fh, 'e')
    # RES related
    y_res = mdp_data.res_penetration(mdp_fh)
    # CO2 related
    y_price = mdp_data.co2_current_price(mdp_fh)
    y_tax = mdp_data.co2_tax_collected(mdp_fh)
    y_emit = mdp_data.co2_emissions(mdp_fh)
    # Cost related
    y_total = mdp_data.cost_total(mdp_fh)
    y_breakdown = mdp_data.cost_breakdown_components(mdp_fh, components)
    y_percents = mdp_data.cost_breakdown_components(mdp_fh, components, is_percent=True)
    # Store data to be used by visualize commands.
    data = OrderedDict()
    data['tech_stage'] = data_dict_array_to_list(y_v)
    data['res_plants'] = data_dict_array_to_list(y_r)
    data['res_penetration'] = data_dict_array_to_list(y_res)
    data['tax_level'] = data_dict_array_to_list(y_l)
    data['tax_adjustment'] = data_dict_array_to_list(y_e)
    data['co2_price'] = data_dict_array_to_list(y_price)
    data['co2_tax'] = data_dict_array_to_list(y_tax)
    data['co2_emissions'] = data_dict_array_to_list(y_emit)
    data['cost_total'] = data_dict_array_to_list(y_total)
    data['cost_breakdown'] = y_breakdown.tolist()
    data['cost_percent'] = y_percents.tolist()
    # One data file per params file.
    data_dir = Path("results/v{}/data".format(DIR_VERSION))
    df = data_dir / "d_v{}_{}.txt".format(DIR_VERSION, args.paramsfile)
    with open(df, 'w+') as datafile:
        datafile.write(str(data))
    datafile.close()


if __name__ == "__main__":
    main(sys.argv[1:])
