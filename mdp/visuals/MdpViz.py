from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


class MdpDataGatherer():
    def __init__(self, mdp_model, n_iter, t_range):
        self.mdp_model = mdp_model
        self.instances = OrderedDict()
        self.n_iter = n_iter
        self.t0 = t_range[0]
        self.tN = t_range[1]

    def add_mdp_instance(self, paramsfile, params):
        assert(self.mdp_model.param_names == list(params.keys()))
        mdp_fh = self.mdp_model.run_fh(params)
        self.instances[paramsfile] = mdp_fh
        return mdp_fh

    ## COST

    # Get all cost components averaged across stochastic tech stage.
    def cost_breakdown_components(self, mdp_fh, components):
        policy_all, avg_techstages = self.aggregate_annotated_policies(mdp_fh)
        y_all = []
        for policy in policy_all:
            y_components = []
            for c in components:
                y = [mdp_fh.mdp_cost.calc_partial_cost(state, a, c) for state, a in policy]
                y_components.append(y)
            y_components = np.stack(np.asarray(y_components), axis=0)
            y_all.append(y_components)
        y_all = np.sum(y_all, axis=0)/self.n_iter
        return y_all

    # Get single cost component averaged across stochastic tech stage.
    def cost_single_component(self, mdp_fh, component):
        policy_all, avg_techstages = self.aggregate_annotated_policies(mdp_fh)
        y_all = []
        for policy in policy_all:
            y_component = [mdp_fh.mdp_cost.calc_partial_cost(state, a, component) for state, a in policy]
            y_all.append(y_component)
        y_all_lower, y_all_upper = self.calc_data_bounds(y_all)
        y_all = np.sum(y_all, axis=0)/self.n_iter
        return y_all

    # Get total cost averaged across stochastic tech stage.
    def cost_total(self, mdp_fh):
        policy_all, avg_techstages = self.aggregate_annotated_policies(mdp_fh)
        y_all = []
        for policy in policy_all:
            y_total = [mdp_fh.mdp_cost.calc_total_cost(state, a) for state, a in policy]
            y_all.append(y_total)
        y_all_lower, y_all_upper = self.calc_data_bounds(y_all)
        y_all = np.sum(y_all, axis=0)/self.n_iter
        return y_all

    ## HELPER FUNCTIONS

    # Aggregate annotated policy for multiple runs of MDP.
    def aggregate_annotated_policies(self, mdp_fh):
        policy_all = []
        runs = self.repeat_mdp_stochastic_techstage(mdp_fh)
        for run in runs:
            # Trim annotated policy to given time range.
            policy_all.append(self.annotate_opt_policy_techstage(mdp_fh, run)[self.t0:self.tN])
        avg_techstages = np.sum(runs, axis=0)[self.t0:self.tN]/self.n_iter
        return policy_all, avg_techstages

    # Zip state with action taken in state for optimal policy of single MDP run.
    def annotate_opt_policy_techstage(self, mdp_fh, run):
        opt_policy = mdp_fh.mdp_inst.policy
        policy_annotated = []
        t = 0
        r = 0
        v = 0
        l = mdp_fh.n_tax_levels // 2
        e = 0
        for step in np.arange(0, mdp_fh.n_years):
            state_curr = (t, v, r, l, e)
            # Get tech stage from stochastic run.
            v = run[step]
            # Get updated tax level and delta emissions target.
            l, e = mdp_fh.update_state_end_of_cycle(state_curr)
            state = (t, v, r, l, e)
            idx = mdp_fh.state_to_id[state]
            a = opt_policy[idx][step]
            policy_annotated.append([(t, v, r, l, e), a])
            t += 1
            r += a
        return policy_annotated

    # Calculate confidence interval given a matrix where each row is a data array.
    def calc_data_bounds(data_all):
        data_lower = data_all.min(1)
        data_upper = data_all.max(1)
        return data_lower, data_upper

    # Run MDP multiple times with stochastically calculated tech stage.
    def repeat_mdp_stochastic_techstage(self, mdp_fh):
        runs = np.zeros([self.n_iter, mdp_fh.n_years], dtype=int)
        for i in np.arange(self.n_iter):
            techstage = 0
            p_adv = mdp_fh.p_adv_tech[0]
            for step in np.arange(1, mdp_fh.n_years):
                # Decide whether or not the tech stage advances for next year.
                adv = np.random.binomial(1, p_adv)
                if adv and techstage < mdp_fh.n_tech_stages - 1:
                    p_adv = mdp_fh.p_adv_tech[techstage+1]
                    techstage += 1
                runs[i][step] = techstage
        return runs
