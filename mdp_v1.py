
import numpy as np
import mdptoolbox as mtb
from scipy.stats import binom

from collections import OrderedDict
import itertools as it


class MdpModelV1():
    def __init__(self):
        self.params_to_policy = OrderedDict()
        self.param_names = ['n_years',
                            'n_tech_stages',
                            'n_plants',
                            'fplant_size',
                            'fplant_capacity',
                            'rplant_capacity',
                            'c_co2_init',
                            'co2_inc',
                            'c_ff_fix',
                            'c_ff_var',
                            'ff_emit',
                            'c_res_cap',
                            'c_bss_cap',
                            'c_bss_fix',
                            'c_bss_var',
                            'p_rplant_fail',
                            'p_adv_tech_stage',
                            'disc_rate']

    def run_param_ranges(self, param_ranges):
        param_combos = it.product(**param_ranges.values())
        for combo in param_combos:
            params = self.create_params(combo)
            mdp_fh = self.run_single(params)
            self.params_to_policy[params] = mdp_fh.mdp_inst.policy

    def run_single(self, params):
        mdp_fh = MdpFiniteHorizonV1(params)
        mdp_fh.initialize()
        mdp_fh.run()
        return mdp_fh

    def print_single(self, mdp_fh):
        assert(mdp_fh is not None)
        mdp_fh.print_params()
        mdp_fh.print_policy()

    def create_params(self, param_list):
        params = OrderedDict()
        for i in np.arange(len(self.param_names)):
            params[self.param_names[i]] = param_list[i]
        return params


class MdpFiniteHorizonV1():
    def __init__(self, params):
        self.mdp_inst = None
        self.params = params
        # Parameters
        self.n_years = params['n_years']
        self.n_tech_stages = params['n_tech_stages']
        self.n_plants = params['n_plants']
        self.fplant_size = params['fplant_size']
        self.fplant_capacity = params['fplant_capacity']
        self.rplant_size = params['fplant_size']*params['fplant_capacity']/params['rplant_capacity']
        self.rplant_capacity = params['rplant_capacity']
        self.c_co2_init = params['c_co2_init']
        self.co2_inc = params['co2_inc']
        self.c_ff_fix = params['c_ff_fix']
        self.c_ff_var = params['c_ff_var']
        self.ff_emit = params['ff_emit']
        self.c_res_cap = params['c_res_cap']
        self.c_bss_cap = params['c_bss_cap']
        self.c_bss_fix = params['c_bss_fix']
        self.c_bss_var = params['c_bss_var']
        self.p_rplant_fail = params['p_rplant_fail']
        self.p_adv_tech_stage = params['p_adv_tech_stage']
        self.disc_rate = params['disc_rate']
        # Dimensions
        self.A = self.n_plants + 1
        self.S = (self.n_years+1) * self.n_tech_stages * (self.n_plants+1)
        # States
        self.state_to_id = OrderedDict()
        self.id_to_state = OrderedDict()
        # Matrices
        self.transitions = None
        self.rewards = None

    def initialize(self):
        print("Initializing MDP v1...\n")
        self._enumerate_states()
        self._trans_probs_wrapper()
        self._rewards_wrapper()
        self.mdp_inst = mtb.mdp.FiniteHorizon(self.transitions,
                                              self.rewards,
                                              self.disc_rate,
                                              self.n_years)
        print("Initialization done.\n")

    def run(self):
        print("Running MDP v1...")
        self.mdp_inst.run()
        print("MDP done.\n")

    def print_params(self):
        print("PARAMETERS:")
        for k, v in self.params.items():
            print(k, ":", v)
        print("\n")

    def print_policy(self):
        assert self.mdp_inst is not None
        print("OPTIMAL POLICY:\nState\t     Time")
        for row, state in zip(self.mdp_inst.policy, self._get_iter_states()):
            print(state, ": ", row)

    def print_rewards(self):
        assert self.mdp_inst is not None
        print("REWARDS MATRIX:")
        for row, state in zip(self.rewards, self._get_iter_states()):
            print(state, ": ", row)

    # STATE SPACE

    def _enumerate_states(self):
        idx = 0
        iter_states = self._get_iter_states()
        for state in iter_states:
            (t, v, r) = state
            self.state_to_id[state] = idx
            idx += 1
        self.id_to_state = {v: k for k, v in self.state_to_id.items()}

    # TRANSITION PROBABILITIES

    def _trans_probs_wrapper(self):
        self.transitions = np.zeros([self.A, self.S, self.S])
        print("Filling transitions probabilities for A = 0 (do nothing)...")
        self._fill_trans_donothing()
        print("Filling transitions probabilities for other A...")
        self._fill_trans_other()
        print("Transitions done.\n")

    def _fill_trans_donothing(self):
        iter_states = self._get_iter_states()
        for state in iter_states:
            (t, v, r), state_curr, idx_curr = self._breakdown_state(state)
            # Edge case for terminal state.
            if t == self.n_years:
                self.transitions[0][idx_curr][idx_curr] = 1.0
                continue
            self._loop_failure(state_curr, 0, 0)
            assert np.isclose(np.sum(self.transitions[0][idx_curr]),
                              1.0), np.sum(self.transitions[0][idx_curr])

    def _fill_trans_other(self):
        iter_states = self._get_iter_states()
        for state in iter_states:
            (t, v, r), state_curr, idx_curr = self._breakdown_state(state)
            # 1 up to number of FF plants remaining may be converted.
            for a in np.arange(1, self.A):
                # Transition doesn't matter for last year as long as it exists.
                if t == self.n_years:
                    self.transitions[a][idx_curr][idx_curr] = 1.0
                    continue
                if a > self.n_plants - r:
                    # If action invalid always build max plants possible.
                    self._loop_failure(state_curr, self.n_plants-r, a)
                else:
                    self._loop_failure(state_curr, a, a)
                assert np.isclose(np.sum(self.transitions[a][idx_curr]),
                                  1.0), np.sum(self.transitions[a][idx_curr])
                self._normalize_trans_row(state_curr, a)

    # REWARDS

    def _rewards_wrapper(self):
        self.rewards = np.zeros([self.S, self.A])
        print("Filling rewards...")
        self._fill_rewards()
        print("Rewards done.\n")

    def _fill_rewards(self):
        for a in np.arange(self.A):
            for s in np.arange(self.S):
                state = self.id_to_state[s]
                idx = self.state_to_id[state]
                # Sanity check for integer id.
                assert(idx == s)
                (t, v, r) = state
                cost = self._calc_total_cost(t, v, r, a)
                # Model reward as negative cost.
                self.rewards[idx][a] = -1 * cost

    # COST FUNCTION

    def _calc_total_cost(self, t, v, r, a):
        if a + r > self.n_plants:
            return np.inf
        carbontax = self.c_co2_init * ((1+self.co2_inc)**t)
        hours_yr = 24*365
        # kW per plant should be the same for RES and FF plants.
        kw_plant = self.rplant_size*self.rplant_capacity
        total_ff_emit = self.ff_emit*kw_plant*hours_yr
        c_om_ff = self.c_ff_fix*kw_plant + self.c_ff_var*kw_plant*hours_yr
        c_fplants = (self.n_plants-a) * (c_om_ff + total_ff_emit*carbontax)
        # Assume RES plants have no O&M costs.
        c_rplants = a*self.c_res_cap[v]*kw_plant
        c_bss = self._calc_bss_cost(v, r, a)
        total = (c_fplants+c_rplants+c_bss)/1e6
        return round(total)

    def _calc_bss_cost(self, v, r, a):
        hours1yr = 24*365
        kw_req = self._calc_bss_kw(r, a)
        c_om_bss = self.c_bss_fix*kw_req + self.c_bss_var*kw_req*hours1yr
        total_c_bss_cap = self.c_bss_cap[v]*kw_req
        return round(c_om_bss+total_c_bss_cap)

    def _calc_bss_kw(self, r, a):
        res_penetration = (r+a) / self.n_plants
        return 0.1357 * np.exp(7.3727*res_penetration) + 0.01

    # HELPER FUNCTIONS

    def _get_iter_states(self):
        return it.product(np.arange(self.n_years+1),
                          np.arange(self.n_tech_stages),
                          np.arange(self.n_plants+1))

    def _breakdown_state(self, state):
        (t, v, r) = state
        state_curr = state
        idx_curr = self.state_to_id[state_curr]
        return ((t, v, r), state_curr, idx_curr)

    def _normalize_trans_row(self, state_curr, a):
        idx_curr = self.state_to_id[state_curr]
        self.transitions[a][idx_curr] = self.transitions[a][idx_curr] / np.sum(self.transitions[a][idx_curr])

    def _loop_failure(self, state, a_actual, a):
        (t, v, r), state_curr, idx_curr = self._breakdown_state(state)
        # Any number of existing RES plants may fail (at end of year).
        for e in np.arange(r+1):
            prob_fail = binom.pmf(e, r, self.p_rplant_fail)
            plants_next = r-e+a_actual
            state_next = (t+1, v, plants_next)
            idx_next = self.state_to_id[state_next]
            if v < self.n_tech_stages - 1:
                state_next_v = (t+1, v+1, plants_next)
                idx_next_v = self.state_to_id[state_next_v]
                # Tech stage may remain the same.
                self.transitions[a][idx_curr][idx_next] = (1.0-self.p_adv_tech_stage) * prob_fail
                # Tech stage may advance (assume only possible to advance by 1).
                self.transitions[a][idx_curr][idx_next_v] = self.p_adv_tech_stage * prob_fail
            else:
                # Tech stage must remain the same.
                self.transitions[a][idx_curr][idx_next] = prob_fail
