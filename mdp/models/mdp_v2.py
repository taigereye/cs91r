from collections import OrderedDict
import itertools as it
import numpy as np
import math
import mdptoolbox as mtb


class MdpModelV2():
    def __init__(self):
        self.params_to_policy = OrderedDict()
        self.param_names = ['n_years',
                            'n_tech_stages',
                            'n_plants',
                            'fplant_size',
                            'fplant_capacity',
                            'rplant_capacity',
                            'rplant_lifetime',
                            'c_co2_init',
                            'co2_inc',
                            'c_ff_fix',
                            'c_ff_var',
                            'ff_emit',
                            'c_res_cap',
                            'bss_coefs',
                            'c_bss_cap',
                            'c_bss_fix',
                            'c_bss_var',
                            'p_adv_tech_stage',
                            'disc_rate']

    def run_param_ranges(self, param_ranges):
        param_combos = it.product(**param_ranges.values())
        for combo in param_combos:
            params = self.create_params(combo)
            mdp_fh = self.run_fh(params)
            self.params_to_policy[params] = mdp_fh.mdp_inst.policy

    def run_fh(self, params):
        mdp_fh = MdpFiniteHorizonV2(params)
        mdp_fh.initialize()
        mdp_fh.run()
        return mdp_fh

    def print_fh(self, mdp_fh):
        assert(mdp_fh is not None)
        mdp_fh.print_params()
        mdp_fh.print_policy()

    def create_params(self, param_list):
        params = OrderedDict()
        for i in np.arange(len(self.param_names)):
            params[self.param_names[i]] = param_list[i]
        return params


class MdpFiniteHorizonV2():
    def __init__(self, params):
        self.mdp_inst = None
        self.params = params
        self.scale_down = 9
        # Parameters
        self.n_years = params['n_years']
        self.n_tech_stages = params['n_tech_stages']
        self.n_plants = params['n_plants']
        self.fplant_size = params['fplant_size']
        self.fplant_capacity = params['fplant_capacity']
        self.rplant_size = params['fplant_size']*params['fplant_capacity']/params['rplant_capacity']
        self.rplant_capacity = params['rplant_capacity']
        self.rplant_lifetime = params['rplant_lifetime']
        self.c_co2_init = params['c_co2_init']
        self.co2_inc = params['co2_inc']
        self.c_ff_fix = params['c_ff_fix']
        self.c_ff_var = params['c_ff_var']
        self.ff_emit = params['ff_emit']
        self.c_res_cap = params['c_res_cap']
        self.bss_coefs = params['bss_coefs']
        self.bss_hrs = 4
        self.c_bss_cap = params['c_bss_cap']
        self.c_bss_fix = params['c_bss_fix']
        self.c_bss_var = params['c_bss_var']
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
        print("\nInitializing MDP v2...\n")
        self._enumerate_states()
        self._trans_probs_wrapper()
        self._rewards_wrapper()
        self.mdp_inst = mtb.mdp.FiniteHorizon(self.transitions,
                                              self.rewards,
                                              self.disc_rate,
                                              self.n_years)
        print("Initialization done.\n")

    def run(self):
        print("Running MDP v2...")
        self.mdp_inst.run()
        print("MDP done.\n")

    def print_params(self):
        print("PARAMETERS:")
        for k, v in self.params.items():
            print("%s: %s" % (k, v))
        print("\n")

    def print_policy(self):
        assert self.mdp_inst is not None
        print("OPTIMAL POLICY:\nState\t     Time")
        self._print_labeled_matrix(self.mdp_inst.policy)

    def print_partial_costs(self, component):
        components = ["rplants_total",
                      "rplants_cap",
                      "rplants_replace",
                      "fplants_total",
                      "fplants_OM",
                      "fplants_OM_fix",
                      "fplants_OM_var",
                      "co2_emit",
                      "co2_tax",
                      "storage_total",
                      "storage_cap",
                      "storage_OM",
                      "storage_OM_fix",
                      "storage_OM_var"]
        if component not in components:
            raise ValueError("Invalid component type. Expected one of %s" % components)
        costs, percents = self._fill_partial_costs(component)
        print("COST MATRIX: %s\nState\t     Time" % component)
        self._print_labeled_matrix(costs, precision=3)
        print("\n\nPERCENTAGE MATRIX: %s\nState\t     Time" % component)
        self._print_labeled_matrix(percents, precision=2)

    def print_rewards(self):
        assert self.mdp_inst is not None
        print("REWARDS MATRIX:")
        self._print_labeled_matrix(self.rewards, precision=3)

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
            # Transition doesn't matter for last year as long as it exists.
            if t == self.n_years:
                self.transitions[0][idx_curr][idx_curr] = 1.0
                continue
            self._single_action_prob(state_curr, 0, 0)
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
                    # If action invalid always build maximum plants possible.
                    self._single_action_prob(state_curr, a, self.n_plants-r)
                else:
                    self._single_action_prob(state_curr, a, a)
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
                if cost < np.inf:
                    cost /= self.scale_down
                # Model reward as negative cost.
                self.rewards[idx][a] = -1 * cost

    # COST FUNCTION

    def _calc_total_cost(self, t, v, r, a):
        if a + r > self.n_plants:
            return np.inf
        f = self.n_plants-(r+a)
        hours_yr = 24*365
        # kW produced per plant should be the same for RES and FF.
        kw_plant = self.fplant_size*self.fplant_capacity
        co2_tax = self.c_co2_init * ((1+self.co2_inc)**t)
        co2_emit = self.ff_emit*kw_plant*hours_yr
        fplants_om_fix = self.c_ff_fix*kw_plant
        fplants_om_var = self.c_ff_var*kw_plant
        fplants_total = f * (fplants_om_fix+fplants_om_var+co2_tax*co2_emit)
        rplants_cap = self.c_res_cap[v]*self.rplant_size
        # Model plant failure as annual O&M cost.
        rplants_replace = rplants_cap/self.rplant_lifetime
        rplants_total = a * rplants_cap + r * rplants_replace
        bss_total = self._calc_bss_cost(v, r, a)
        cost = fplants_total + rplants_total + bss_total
        return cost

    def _calc_bss_cost(self, v, r, a):
        # Additional storage capacity needed as percentage of total system load.
        kwh_storage = self._calc_bss_kwh(r, a) - self._calc_bss_kwh(r, 0)
        storage_om = self.c_bss_fix*kwh_storage*self.bss_hrs + self.c_bss_var*kwh_storage
        storage_cap = self.c_bss_cap[v]*kwh_storage
        return storage_om+storage_cap

    # HELPER FUNCTIONS

    def _breakdown_state(self, state):
        (t, v, r) = state
        state_curr = state
        idx_curr = self.state_to_id[state_curr]
        return ((t, v, r), state_curr, idx_curr)

    def _calc_bss_kwh(self, r, a):
        hours_yr = 24*365
        kw_sys_total = self.fplant_size*self.fplant_capacity * self.n_plants
        res_percent = (r+a)*100 / self.n_plants
        bss_percent = self.bss_coefs[0] * np.exp(self.bss_coefs[1]*res_percent) + self.bss_coefs[2]
        return bss_percent/100*kw_sys_total*hours_yr

    def _calc_partial_cost(self, t, v, r, a, component):
        cost = 0
        f = self.n_plants-(r+a)
        if f < 0:
            return np.inf
        hours_yr = 24*365
        kw_plant = self.fplant_size*self.fplant_capacity
        co2_tax = self.c_co2_init * ((1+self.co2_inc)**t)
        co2_emit = self.ff_emit*kw_plant*hours_yr
        fplants_om_fix = self.c_ff_fix*kw_plant
        fplants_om_var = self.c_ff_var*kw_plant
        rplants_cap = self.c_res_cap[v]*self.rplant_size
        rplants_replace = rplants_cap/self.rplant_lifetime
        kwh_storage = self._calc_bss_kwh(r, a) - self._calc_bss_kwh(r, 0)
        storage_cap = self.c_bss_cap[v]*kwh_storage
        storage_om_fix = self.c_bss_fix*kwh_storage*self.bss_hrs
        storage_om_var = self.c_bss_var*kwh_storage
        if component == "rplants_total":
            cost = a * rplants_cap + r * rplants_replace
        elif component == "rplants_cap":
            cost = a * rplants_cap
        elif component == "rplants_replace":
            cost = r * rplants_replace
        elif component == "fplants_total":
            cost = f * (fplants_om_fix + fplants_om_var + co2_tax*co2_emit)
        elif component == "fplants_OM":
            cost = f * (fplants_om_fix + fplants_om_var)
        elif component == "fplants_OM_fix":
            cost = f * (fplants_om_fix)
        elif component == "fplants_OM_var":
            cost = f * (fplants_om_fix)
        elif component == "co2_emit":
            cost = f * (co2_emit)
        elif component == "co2_tax":
            cost = f * (co2_emit * co2_tax)
        elif component == "storage_total":
            cost = storage_cap + storage_om_fix + storage_om_var
        elif component == "storage_cap":
            cost = storage_cap
        elif component == "storage_OM":
            cost = storage_om_fix + storage_om_var
        elif component == "storage_OM_fix":
            storage_om_fix
        elif component == "storage_OM_var":
            storage_om_var
        return cost

    def _fill_partial_costs(self, component):
        self._enumerate_states()
        costs = np.zeros([self.S, self.A])
        percents = np.zeros([self.S, self.A])
        for a in np.arange(self.A):
            for s in np.arange(self.S):
                state = self.id_to_state[s]
                idx = self.state_to_id[state]
                # Sanity check for integer id.
                assert(idx == s)
                (t, v, r) = state
                costs[idx][a] = self._calc_partial_cost(t, v, r, a, component)
                # Keep cost positive to save printing space.
                if costs[idx][a] == np.inf:
                    percents[idx][a] = np.inf
                else:
                    costs[idx][a] /= self.scale_down
                    percents[idx][a] = (costs[idx][a]*100) / self._calc_total_cost(t, v, r, a)
        return costs, percents

    def _get_iter_states(self):
        return it.product(np.arange(self.n_years+1),
                          np.arange(self.n_tech_stages),
                          np.arange(self.n_plants+1))

    def _normalize_trans_row(self, state_curr, a):
        idx_curr = self.state_to_id[state_curr]
        self.transitions[a][idx_curr] = self.transitions[a][idx_curr] / np.sum(self.transitions[a][idx_curr])

    def _print_labeled_matrix(self, matrix, linewidth=300, precision=3):
        np.set_printoptions(linewidth=linewidth, precision=precision, suppress=True, floatmode="fixed")
        for row, state in zip(matrix, self._get_iter_states()):
            (t, v, r) = state
            print("({:02d},{:d},{:02d}) : ".format(t, v, r), end="")
            print(row)

    def _round_sig_figs(self, num, sig_figs):
        if num == 0:
            return num
        elif num == np.inf:
            return np.inf
        else:
            return round(num, -1*int(math.floor(math.log10(abs(num))) - (sig_figs-1)))

    def _single_action_prob(self, state, a, a_actual):
        (t, v, r), state_curr, idx_curr = self._breakdown_state(state)
        state_next = (t+1, v, r+a_actual)
        idx_next = self.state_to_id[state_next]
        if v < self.n_tech_stages - 1:
            state_next_v = (t+1, v+1, r+a_actual)
            idx_next_v = self.state_to_id[state_next_v]
            # Tech stage may remain the same.
            self.transitions[a][idx_curr][idx_next] = 1.0 - self.p_adv_tech_stage
            # Tech stage may advance (assume only possible to advance by 1).
            self.transitions[a][idx_curr][idx_next_v] = self.p_adv_tech_stage
        else:
            # Tech stage must remain the same.
            self.transitions[a][idx_curr][idx_next] = 1.0
