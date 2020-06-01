import argparse

from pathlib import Path

from mdp.models.MdpV4 import MdpModelV4


class MdpArgs():
    def __init__(self, description):
        self.parser = argparse.ArgumentParser(description=description)
        self.args = None

    def add_confidence_interval(self):
        self.parser.add_argument("--CI", help="plot confidence intervals for single line plots", action='store_true')

    def add_granular(self):
        self.parser.add_argument("--granular", help="plot more granular cost component breakdown", action='store_true')

    def add_emissions_target(self):
        self.parser.add_argument("-e", "--targetsfile", help="txt file with CO2 emissions targets as list")

    def add_iterations(self):
        self.parser.add_argument("-i", "--iterations", help="number of times MDP run with stochastic tech stage", type=int, default=200)

    def add_model_version(self):
        self.parser.add_argument("-m", "--version", help="MDP model version", type=int)

    def add_paramfile_multiple(self):
        self.parser.add_argument("-p", "--paramsfiles", help="list of txt files with version specific params as dict", nargs='*', action='store')

    def add_paramfile_single(self):
        self.parser.add_argument("-p", "--paramsfile", help="txt file with version specific params as dict")

    def add_save(self):
        self.parser.add_argument("--save", help="save plots as png files", action='store_true')

    def add_time_range(self):
        self.parser.add_argument("-t", "--timerange", help="plot only subset of model time horizon", nargs=2, type=int, default=None)

    def check_version(self, v_min, v_max, err_msg):
        if int(self.args.version) < v_min or int(self.args.version) > v_max:
            print("error: {}.".format(err_msg))
            return False
        return True

    def check_paramfile_multiple(self, err_msg):
        if len(self.args.paramsfiles) == 0:
            print("error: {}.".format(err_msg))
            return False
        return True

    def get_args(self):
        self.args = self.parser.parse_args()
        return self.args


def adjust_emissions_target_timeline(targets, cycle, t_range):
    targets_adjusted = []
    for i in range(len(targets)):
        targets_adjusted.append(targets[i])
        targets_adjusted.extend([targets[i] for j in range(cycle)])
    return targets_adjusted[t_range[0]:t_range[1]]


def get_emissions_target(version, targetsfile):
    targets_dir = Path("visuals/v{}/targets".format(version))
    tf = targets_dir / "e_v{}_{}.txt".format(version, targetsfile)
    with open(tf, 'r') as targetsfile:
        emit_targets = eval(targetsfile.read())
    targetsfile.close()
    return emit_targets


def get_mdp_model(version, params_all):
    mdp_model = None
    if int(version) == 4:
        mdp_model = MdpModelV4()
    assert(mdp_model is not None)
    for params in params_all:
        assert(mdp_model.param_names == list(params.keys()))
    return mdp_model


def get_mdp_instance_multiple(mdp_model, params_all):
    mdp_fh_all = []
    for params in params_all:
        mdp_fh = mdp_model.run_fh(params)
        mdp_fh_all.append(mdp_fh)
    return mdp_fh_all


def get_mdp_instance_single(mdp_model, params):
    return mdp_model.run_fh(params)


def get_params_single(version, paramsfile):
    params_dir = Path("results/v{}/params".format(version))
    pf = params_dir / "p_v{}_{}.txt".format(version, paramsfile)
    with open(pf, 'r') as paramsfile:
        params = eval(paramsfile.read())
    paramsfile.close()
    return params


def get_params_multiple(version, paramsfiles):
    params_all = []
    for paramsfile in paramsfiles:
        params_dir = Path("results/v{}/params".format(version))
        pf = params_dir / "p_v{}_{}.txt".format(version, paramsfile)
        with open(pf, 'r') as paramsfile:
            params = eval(paramsfile.read())
        paramsfile.close()
        params_all.append(params)
    return params_all


def get_time_range(args, mdp_fh):
    if args.timerange:
        t0, tN = args.timerange
        t0 = max(0, t0-1)
        if tN - t0 > mdp_fh.n_years:
            print("error: time range {}-{} out of range: {}".format(t0, tN, mdp_fh.n_tech_stages))
            return None
    else:
        t0 = 0
        tN = mdp_fh.n_years
    return [t0, tN]
