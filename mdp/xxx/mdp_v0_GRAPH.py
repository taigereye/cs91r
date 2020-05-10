from mdp_v0 import MdpModelV0

import numpy as np
import matplotlib.pyplot as plt

import sys, os


def graph_by_techstage(mdp_instance, v):
    state_policy = list(zip(mdp_instance.get_iter_states(),
                            mdp_instance.mdp_fh.policy))
    state_policy_v = [sp for sp in state_policy if sp[0][1] == v]


def main():
    mdp_model = MdpModelV0()
    param_list = [50, 3, 10, 600000, 0.35, 100, 0.05, (1284, 746, 456), 68.8, 1.65, 1/25, 0.25, 0.06]
    params = mdp_model.create_params(param_list)
    sys.stdout = open(os.devnull, 'w')  # BLOCK PRINT
    mdp_instance = mdp_model.run_single(params)
    sys.stdout = sys.__stdout__  # ENABLE PRINT


if __name__ == "__main__":
    main()
