from collections import OrderedDict
import itertools as it
import numpy as np
import mdptoolbox as mtb
from scipy.stats import binom


class MdpModelV0():
    def __init__(self):
        self.params_to_policy = OrderedDict()
        self.param_names = ['n_years',
                            'n_tech_stages',
                            'n_plants',
                            'plant_size',
                            'plant_capacity',
                            'c_co2_init',
                            'co2_inc',
                            'c_cap_res',
                            'c_om_ff',
                            'ff_emit',
                            'p_rplant_fail',
                            'p_adv_tech_stage',
                            'disc_rate']

    def run_param_ranges(self, param_ranges):
        param_combos = it.product(param_ranges['n_years'],
                                  param_ranges['n_tech_stages'],
                                  param_ranges['n_plants'],
                                  param_ranges['plant_size'],
                                  param_ranges['plant_capacity'],
                                  param_ranges['c_co2_init'],
                                  param_ranges['co2_inc'],
                                  param_ranges['c_cap_res'],
                                  param_ranges['c_om_ff'],
                                  param_ranges['ff_emit'],
                                  param_ranges['p_rplant_fail'],
                                  param_ranges['p_adv_tech_stage'],
                                  param_ranges['disc_rate'])
        for combo in param_combos:
            params = OrderedDict()
            params['n_years'] = combo[0]
            params['n_tech_stages'] = combo[1]
            params['n_plants'] = combo[2]
            params['plant_size'] = combo[3]
            params['plant_capacity'] = combo[4]
            params['c_co2_init'] = combo[5]
            params['co2_inc'] = combo[6]
            params['c_cap_res'] = combo[7]
            params['c_om_ff'] = combo[8]
            params['ff_emit'] = combo[9]
            params['p_rplant_fail'] = combo[10]
            params['p_adv_tech_stage'] = combo[11]
            params['disc_rate'] = combo[12]
            mdp_instance = self.run_single(params)
            self.params_to_policy[params] = mdp_instance.policy

    def run_single(self, params):
        mdp_instance = MdpFiniteHorizonV0(params)
        mdp_instance.initialize()
        mdp_instance.run()
        return mdp_instance

    def print_single(self, mdp_instance):
        assert(mdp_instance is not None)
        mdp_instance.print_params()
        mdp_instance.print_policy()

    def create_params(self, param_list):
        params = OrderedDict()
        params['n_years'] = param_list[0]
        params['n_tech_stages'] = param_list[1]
        params['n_plants'] = param_list[2]
        params['plant_size'] = param_list[3]
        params['plant_capacity'] = param_list[4]
        params['c_co2_init'] = param_list[5]
        params['co2_inc'] = param_list[6]
        params['c_cap_res'] = param_list[7]
        params['c_om_ff'] = param_list[8]
        params['ff_emit'] = param_list[9]
        params['p_rplant_fail'] = param_list[10]
        params['p_adv_tech_stage'] = param_list[11]
        params['disc_rate'] = param_list[12]
        return params


class MdpFiniteHorizonV0():
    def __init__(self, params):
        self.mdp_fh = None
        self.params = params
        # Parameters
        self.n_years = params['n_years']
        self.n_tech_stages = params['n_tech_stages']
        self.n_plants = params['n_plants']
        self.plant_size = params['plant_size']
        self.plant_capacity = params['plant_capacity']
        self.c_co2_init = params['c_co2_init']
        self.co2_inc = params['co2_inc']
        self.c_cap_res = params['c_cap_res']
        self.c_om_ff = params['c_om_ff']
        self.ff_emit = params['ff_emit']
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
        print("Initializing MDP v0...\n")
        self._enumerate_states()
        self._trans_probs_wrapper()
        self._rewards_wrapper()
        self.mdp_fh = mtb.mdp.FiniteHorizon(self.transitions, self.rewards,
                                            self.disc_rate, self.n_years)
        print("Initialization done.\n")

    def run(self):
        print("Running MDP v0...")
        self.mdp_fh.run()
        print("MDP done.\n")

    def print_params(self):
        print("PARAMETERS:")
        print("n_years:", self.n_years)
        print("n_tech_stages", self.n_tech_stages)
        print("n_plants", self.n_plants)
        print("plant_size:", self.plant_size)
        print("plant_capacity:", self.plant_capacity)
        print("c_co2_init:", self.c_co2_init)
        print("co2_inc:", self.co2_inc)
        print("c_cap_res:", self.c_cap_res)
        print("c_om_ff:", self.c_om_ff)
        print("ff_emit:", self.ff_emit)
        print("p_rplant_fail:", self.p_rplant_fail)
        print("p_adv_tech_stage:", self.p_adv_tech_stage)
        print("disc_rate:", self.disc_rate, "\n")

    def print_policy(self):
        assert self.mdp_fh is not None
        print("OPTIMAL POLICY:\nState\t     Time")
        for row, state in zip(self.mdp_fh.policy, self.get_iter_states()):
            print("%s: [%s]" % (state, ''.join('%03s' % i for i in row)))

    # STATE SPACE

    def _enumerate_states(self):
        idx = 0
        iter_states = self.get_iter_states()
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
        iter_states = self.get_iter_states()
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
        iter_states = self.get_iter_states()
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
                cost = self._calc_cost(t, v, r, a)
                # Model reward as negative cost.
                self.rewards[idx][a] = -1 * cost

    def _calc_cost(self, t, v, r, a):
        if a + r > self.n_plants:
            return np.inf
        carbontax = self.c_co2_init * ((1+self.co2_inc) ** t)
        hoursyr = 24*52*365
        cost_ff_emit = self.ff_emit*self.plant_size*self.plant_capacity*hoursyr*carbontax
        cost_fplants = (self.n_plants-a) * (self.c_om_ff*self.plant_size + cost_ff_emit)
        # Assume renewable plants cost nothing after construction.
        cost_rplants = a*self.c_cap_res[v]*self.plant_size
        total = (cost_rplants+cost_fplants) / 1e6
        return round(total)

    # HELPER FUNCTIONS

    def get_iter_states(self):
        return it.product(np.arange(self.n_years+1),
                          np.arange(self.n_tech_stages),
                          np.arange(self.n_plants+1))

    def _breakdown_state(self, state):
        (t, v, r) = state
        state_curr = state
        idx_curr = self.state_to_id[state_curr]
        return ((t, v, r), state_curr, idx_curr)

    def _loop_failure(self, state, a_actual, a):
        (t, v, r), state_curr, idx_curr = self._breakdown_state(state)
        # Any number of existing renewable plants may fail (at end of year).
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

    def _normalize_trans_row(self, state_curr, a):
        idx_curr = self.state_to_id[state_curr]
        self.transitions[a][idx_curr] = self.transitions[a][idx_curr] / np.sum(self.transitions[a][idx_curr])
