from mdp_v1 import MdpModelV1


def main():
    mdp_model = MdpModelV1()
    param_list = [50, 3, 10, 600000, 0.6, 0.3, 40, 0.05, 26.27, 0.043, 0.0011, (1284, 746, 456), 0, 1/25, 0.25, 0.06]
    params = mdp_model.create_params(param_list)
    mdp_fh = mdp_model.run_single(params)
    mdp_model.print_single(mdp_fh)


if __name__ == "__main__":
    main()
