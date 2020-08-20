import sys

from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import mdp.analysis.MdpCLI as cl
from mdp.analysis.MdpCLI import MdpArgs
from mdp.analysis.MdpViz import MdpDataGatherer, MdpPlotter


og = sys.stdout


def main(argv):
    parser = MdpArgs(description="debug")
    parser.add_model_version()
    parser.add_paramsfile_single()
    args = parser.get_args()

    params = cl.get_params_single(args.version, args.paramsfile)
    mdp_model = cl.get_mdp_model(args.version, [params])
    mdp_fh = cl.get_mdp_instance_single(mdp_model, params)

    # Iterations set to 1, timerange set to 0-20 yr.
    mdp_data = MdpDataGatherer(mdp_model, 1, [0, 20])
    # Variable pulled out of _adjust_co2_tax function in MdpCostCalculator.
    y_base = mdp_data.co2_base(mdp_fh)
    # Variable pulled from 
    y_l = mdp_data.get_state_variable(mdp_fh, 'l')
    y_price = mdp_data.co2_current_price(mdp_fh)
    y_e = mdp_data.get_state_variable(mdp_fh, 'e')
    y_inc = mdp_data.co2_inc(mdp_fh)

    # # Use if changing iterations to > 1.
    # y_base = np.mean(mdp_data.co2_base(mdp_fh), axis=0)
    # y_l = np.mean(mdp_data.get_state_variable(mdp_fh, 'l'), axis=0)
    # y_price = np.mean(mdp_data.co2_current_price(mdp_fh), axis=0)
    # y_e = np.mean(mdp_data.get_state_variable(mdp_fh, 'e'), axis=0)
    # y_inc = np.mean(mdp_data.co2_inc(mdp_fh), axis=0)

    np.set_printoptions(linewidth=200)

    

    with open("debug_co2_output.txt", "a+") as debugfile:
        sys.stdout = debugfile
        print("### DEBUGGING CO2 ###\n\n")
        print(params, "\n")
        time = 0
        # for l, e, base, inc in zip(y_l, y_e, y_base, y_inc):
        #     print("TIME: ", time)
        #     print("level: ", l)
        #     print("act: ", e)
        #     print("base: ", base)
        #     print("inc: ", inc)
        #     time += 1
        print("inc: ", y_inc)
        print("base: ", y_base)
        print("level: ", y_l)
        print("price: ", y_price)
        print("act: ", y_e)
        print("\n\n")
        sys.stdout = og
    debugfile.close()

    
    

if __name__ == "__main__":
    main(sys.argv[1:])