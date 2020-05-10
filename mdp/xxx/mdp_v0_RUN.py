from mdp_v0 import MdpModelV0


def main():
    mdp_model = MdpModelV0()
    param_list = [50, 3, 10, 600000, 0.35, 100, 0.05, (1284, 746, 456), 68.8, 1.65, 1/25, 0.25, 0.06]
    params = mdp_model.create_params(param_list)
    mdp_fh = mdp_model.run_single(params)
    mdp_model.print_single(mdp_fh)


if __name__ == "__main__":
    main()
