
import numpy as np
import mdptoolbox as mtb
from scipy.stats import binom

from collections import OrderedDict
import itertools as it


class MdpModel():
    def __init__(self):
        self.params_to_policy = OrderedDict()

    def run_param_ranges(self, param_ranges, disc_rates):
        all_n_years = param_ranges['n_years']
        all_n_techstages = param_ranges['n_techstages']
        all_n_plants = param_ranges['n_plants']
        all_c_co2_init = param_ranges['c_co2_init']
        all_co2_inc = param_ranges['co2_inc']
        all_c_cap_res = param_ranges['c_cap_res']
        all_c_om_ff = param_ranges['c_om_ff']
        all_ff_emit = param_ranges['ff_emit']
        all_p_rplant_fail = 1 / param_ranges['rplant_life']
        all_p_adv_techstage = param_ranges['p_adv_techstage']
        param_combos = it.product(all_n_years, all_n_techstages, all_n_plants,
                                  all_c_co2_init, all_co2_inc, all_c_cap_res,
                                  all_c_om_ff, all_ff_emit, all_p_rplant_fail, 
                                  all_p_adv_techstage, disc_rates)
        for combo in param_combos:
            params = OrderedDict()
            params['n_years'] = combo[0]
            params['n_techstages'] = combo[1]
            params['n_plants'] = combo[2]
            params['c_co2_init'] = combo[3]
            params['co2_inc'] = combo[4]
            params['c_cap_res'] = combo[5]
            params['c_om_ff'] = combo[6]
            params['ff_emit'] = combo[7]
            params['p_rplant_fail'] = combo[8]
            params['p_adv_techstage'] = combo[9]
            disc_rate = combo[10]
            mdp_instance = self.run_single(params, disc_rate)
            self.params_to_policy[params] = mdp_instance.policy

    def run_single(self, params):
        mdp_instance = MdpFiniteHorizon(params)
        mdp_instance.initialize()
        mdp_instance.run()
        return mdp_instance

    def print(self):
        assert(self.mdp_instance is not None)
        print("Parameters:\n", self.mdp_instance.params)
        self.mdp_instance.print_policy()


class MdpFiniteHorizon():
    def __init__(self, params, disc_rate):
        self.mdp_fh = None
        # Parameters
        self.n_years = self.params['n_years']
        self.n_techstages = self.params['n_techstages']
        self.n_plants = self.params['n_plants']
        self.c_co2_init = self.params['c_co2_init']
        self.co2_inc = self.params['co2_inc']
        self.c_cap_res = self.params['c_cap_res']
        self.c_om_ff = self.params['c_om_ff']
        self.ff_emit = self.params['ff_emit']
        self.p_rplant_fail = 1 / self.params['rplant_life']
        self.p_adv_techstage = params['p_adv_techstage']
        self.disc_rate = disc_rate
        # Dimensions
        self.A = self.n_plants + 1
        self.S = (self.n_years+1) * self.n_techstages * (self.n_plants+1)
        # States
        self.state_to_id = OrderedDict()
        self.id_to_state = OrderedDict()
        # Matrices
        self.transitions = None
        self.rewards = None

    def initialize(self):
        print("Initializing MDP...")
        self._enumerate_states()
        self._trans_probs_wrapper()
        self._trans_probs_wrapper()
        self._rewards_wrapper()
        self.mdp_fh = mtb.mdp.FiniteHorizon(self.transitions, self.rewards, self.disc_rate, self.n_years)
        print("Initialization done.\n")

    def run(self):
        print("Running MDP...")
        self.mdp_fh.run()
        print("MDP done.\n")

    def print_policy(self):
        assert self.mdp_fh is not None
        print("Optimal policy:\nState\t     Time")
        for row, state in zip(self.mdp_fh.policy, self._get_iter_states()):
            print(state, ": ", row)

    # STATE SPACE

    def _enumerate_states(self):
        idx = 0
        iter_states = self._get_iter_states()
        for state in iter_states:
            (t, v, r) = state
            self.self.state_to_id[state] = idx
            idx += 1
        self.id_to_state = {v: k for k, v in self.state_to_id.items()}

    # TRANSITION PROBABILITIES

    def _trans_probs_wrapper(self):
        self.transitions = np.zeros([self.A, self.S, self.S])
        print("Filling transitions probabilities for A = 0 (do nothing)...")
        self._fill_trans_donothing(0)
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
            # 1 up to number of fossil fuel plants remaining may be converted.
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
                self.normalize_trans_row(state_curr, a)

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
                cost = self.calc_cost(t, v, r, a)
                # Model reward as negative cost.
                self.rewards[idx][a] = -1 * cost

    def _calc_cost(self, t, v, r, a):
        if a + r > self.n_plants:
            return np.inf
        carbontax = self.c_co2_init * (1.05 ** t)
        cost_fplants = (self.n_plants - a) * (self.c_om_ff + self.ff_emit * carbontax)
        # Assume renewable plants cost nothing after construction.
        cost_rplants = a * self.c_cap_res[v]
        total = cost_rplants + cost_fplants
        return round(total)

    # HELPER FUNCTIONS

    def _get_iter_states(self):
        return it.product(np.arange(self.n_years+1),
                          np.arange(self.n_techstages),
                          np.arange(self.n_plants+1))

    def __breakdown_state(self, state):
        (t, v, r) = state
        state_curr = state
        idx_curr = self.state_to_id[state_curr]
        return ((t, v, r), state_curr, idx_curr)

    def _normalize_trans_row(self, state_curr, a):
        idx_curr = self.state_to_id[state_curr]
        self.transitions[a][idx_curr] = self.transitions[a][idx_curr] / np.sum(self.transitions[a][idx_curr])

    def _loop_failure(self, state, a_actual, a):
        (t, v, r), state_curr, idx_curr = self._breakdown_state(state)
        # Any number of existing renewable plants may fail (at end of year).
        for e in np.arange(r+1):
            prob_fail = binom.pmf(e, r, self.p_rplant_fail)
            plants_next = r-e+a_actual
            state_next = (t+1, v, plants_next)
            idx_next = self.state_to_id[state_next]
            if v < self.n_techstages - 1:
                state_next_v = (t+1, v+1, plants_next)
                idx_next_v = self.state_to_id[state_next_v]
                # Tech stage may remain the same.
                self.transitions[a][idx_curr][idx_next] = (1.0-self.p_adv_techstage) * prob_fail
                # Tech stage may advance (assume only possible to advance by 1).
                self.transitions[a][idx_curr][idx_next_v] = self.p_adv_techstage * prob_fail
            else:
                # Tech stage must remain the same.
                self.transitions[a][idx_curr][idx_next] = prob_fail
