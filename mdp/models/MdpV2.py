from collections import OrderedDict
import itertools as it
import numpy as np
import mdptoolbox as mtb


class MdpModelV2():
    def __init__(self):
        self.params_to_policy = OrderedDict()
        self.param_names = ['n_years',
                            'n_tech_stages',
                            'n_plants',
                            'ff_size',
                            'ff_capacity',
                            'ff_lifetime',
                            'res_capacity',
                            'res_lifetime',
                            'c_co2_init',
                            'c_co2_inc',
                            'c_ff_cap',
                            'c_ff_fix',
                            'c_ff_var',
                            'ff_emit',
                            'c_res_cap',
                            'storage_mix',
                            'storage_coefs',
                            'bss_hrs',
                            'c_bss_cap',
                            'c_bss_fix',
                            'c_bss_var',
                            'c_phs_cap',
                            'c_phs_fix',
                            'p_adv_tech',
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
        print("\n\n")
        mdp_fh.print_policy()
        print("\n\n")
        mdp_fh.print_rewards()

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
        # Cost
        self.mdp_cost = MdpCostCalculatorV2(params)
        # Parameters
        self.n_years = params['n_years']
        self.n_tech_stages = params['n_tech_stages']
        self.n_plants = params['n_plants']
        self.p_adv_tech = params['p_adv_tech']
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
                                              1-self.disc_rate,
                                              self.n_years)
        print("Initialization done.\n")

    def run(self):
        print("Running MDP v2...")
        self.mdp_inst.run()
        print("MDP done.\n")

    def print_params(self):
        print("PARAMETERS:\n")
        for k, v in self.params.items():
            print("%s: %s" % (k, v))

    def print_policy(self):
        assert self.mdp_inst is not None
        print("OPTIMAL POLICY:\n\nState\t     Time")
        self._print_labeled_matrix(self.mdp_inst.policy)

    def print_partial_costs(self, component):
        components = ["co2_emit",
                      "co2_tax",
                      "ff_total",
                      "ff_replace",
                      "ff_om",
                      "res_total",
                      "res_cap",
                      "res_replace",
                      "bss_total",
                      "bss_cap",
                      "bss_om",
                      "phs_total",
                      "phs_cap",
                      "phs_om",
                      "storage_total",
                      "storage_cap",
                      "storage_om"]
        if component not in components:
            raise ValueError("Invalid component type. Expected one of {}".format(components))
        costs, percents = self._fill_partial_costs(component)
        print("COST MATRIX: %s\n\nState\t     Time" % component)
        self._print_labeled_matrix(costs, to_round=True)
        print("\n\nPERCENTAGE MATRIX: %s\n\nState\t     Time" % component)
        self._print_labeled_matrix(percents, precision=2)

    def print_rewards(self):
        assert self.mdp_inst is not None
        print("REWARDS MATRIX:")
        self._print_labeled_matrix(self.rewards, to_round=True)

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
                cost = self.mdp_cost.calc_total_cost(t, v, r, a)
                if cost < np.inf:
                    cost /= self.scale_down
                # Model reward as negative cost.
                self.rewards[idx][a] = -1 * cost

    # HELPER FUNCTIONS

    def _breakdown_state(self, state):
        (t, v, r) = state
        state_curr = state
        idx_curr = self.state_to_id[state_curr]
        return ((t, v, r), state_curr, idx_curr)

    def _fill_partial_costs(self, component):
        self._enumerate_states()
        mdp_cost = MdpCostCalculatorV2(self.params)
        costs = np.zeros([self.S, self.A])
        percents = np.zeros([self.S, self.A])
        for a in np.arange(self.A):
            for s in np.arange(self.S):
                state = self.id_to_state[s]
                idx = self.state_to_id[state]
                # Sanity check for integer id.
                assert(idx == s)
                (t, v, r) = state
                costs[idx][a] = mdp_cost.calc_partial_cost(t, v, r, a, component)
                # Keep cost positive to save printing space.
                if costs[idx][a] == np.inf:
                    percents[idx][a] = np.inf
                else:
                    costs[idx][a] /= self.scale_down
                    total_cost = mdp_cost.calc_total_cost(t, v, r, a)
                    if total_cost == 0:
                        percents[idx][a] = 0
                    else:
                        percents[idx][a] = (costs[idx][a]*100) / total_cost
        return costs, percents

    def _get_iter_states(self):
        return it.product(np.arange(self.n_years+1),
                          np.arange(self.n_tech_stages),
                          np.arange(self.n_plants+1))

    def _normalize_trans_row(self, state_curr, a):
        idx_curr = self.state_to_id[state_curr]
        self.transitions[a][idx_curr] = self.transitions[a][idx_curr] / np.sum(self.transitions[a][idx_curr])

    def _print_labeled_matrix(self, matrix, to_round=False, linewidth=300, precision=3):
        np.set_printoptions(linewidth=linewidth, precision=precision, floatmode='maxprec')
        for row, state in zip(matrix, self._get_iter_states()):
            (t, v, r) = state
            print("({:02d},{:d},{:02d}) : ".format(t, v, r), end="")
            if to_round:
                print(np.array([round(i) for i in row]))
            else:
                print(row)

    def _single_action_prob(self, state, a, a_actual):
        (t, v, r), state_curr, idx_curr = self._breakdown_state(state)
        state_next = (t+1, v, r+a_actual)
        idx_next = self.state_to_id[state_next]
        if v < self.n_tech_stages - 1:
            state_next_v = (t+1, v+1, r+a_actual)
            idx_next_v = self.state_to_id[state_next_v]
            # Tech stage may remain the same.
            self.transitions[a][idx_curr][idx_next] = 1.0 - self.p_adv_tech
            # Tech stage may advance (assume only possible to advance by 1).
            self.transitions[a][idx_curr][idx_next_v] = self.p_adv_tech
        else:
            # Tech stage must remain the same.
            self.transitions[a][idx_curr][idx_next] = 1.0


class MdpCostCalculatorV2():
    def __init__(self, params):
        self.params = params
        self.n_plants = params['n_plants']
        # CO2
        self.c_co2_init = params['c_co2_init']
        self.c_co2_inc = params['c_co2_inc']
        # FF
        self.ff_size = params['ff_size']
        self.ff_capacity = params['ff_capacity']
        self.ff_lifetime = params['ff_lifetime']
        self.c_ff_cap = params['c_ff_cap']
        self.c_ff_fix = params['c_ff_fix']
        self.c_ff_var = params['c_ff_var']
        self.ff_emit = params['ff_emit']
        # RES
        self.res_size = params['ff_size']*params['ff_capacity']/params['res_capacity']
        self.res_capacity = params['res_capacity']
        self.res_lifetime = params['res_lifetime']
        self.c_res_cap = params['c_res_cap']
        # Storage
        self.storage_mix = params['storage_mix']
        self.storage_coefs = params['storage_coefs']
        self.bss_hrs = params['bss_hrs']
        self.c_bss_cap = params['c_bss_cap']
        self.c_bss_fix = params['c_bss_fix']
        self.c_bss_var = params['c_bss_var']
        self.c_phs_cap = params['c_phs_cap']
        self.c_phs_fix = params['c_phs_fix']

    def calc_partial_cost(self, t, v, r, a, component):
        cost = 0
        f = self.n_plants-(r+a)
        if f < 0:
            return np.inf
        if component == "co2_emit":
            cost = self.co2_emit(f)
        elif component == "co2_tax":
            cost = self.co2_tax(t, f)
        elif component == "ff_total":
            cost = self._ff_total(f)
        elif component == "ff_om":
            cost = self._ff_om(f)
        elif component == "ff_replace":
            cost = self._ff_replace(f)
        elif component == "res_total":
            cost = self._res_total(v, r, a)
        elif component == "res_cap":
            cost = self._res_cap(v, a)
        elif component == "res_replace":
            cost = self._res_replace(v, r)
        elif component == "bss_total":
            cost = self._bss_total(v, r, a)
        elif component == "bss_cap":
            cost = self._bss_cap(v, r, a)
        elif component == "bss_om":
            cost = self._bss_om(r, a)
        elif component == "phs_total":
            cost = self._phs_total(v, r, a)
        elif component == "phs_cap":
            cost = self._phs_cap(v, r, a)
        elif component == "phs_om":
            cost = self._phs_om(r, a)
        elif component == "storage_total":
            cost = self._storage_total(v, r, a)
        elif component == "storage_cap":
            cost = self._storage_cap(v, r, a)
        elif component == "storage_om":
            cost = self._storage_om(r, a)
        return cost

    def calc_total_cost(self, t, v, r, a):
        if a + r > self.n_plants:
            return np.inf
        f = self.n_plants - (r+a)
        co2_tax = self.co2_tax(t, f)
        ff_total = self._ff_total(f)
        res_total = self._res_total(v, r, a)
        storage_total = self._storage_total(v, r, a)
        return co2_tax + ff_total + res_total + storage_total

    # FOSSIL FUEL PLANTS

    def _ff_replace(self, f):
        return f * (self.c_ff_cap*self.ff_size/self.ff_lifetime)

    def _ff_om(self, f):
        kw_plant = self.ff_size*self.ff_capacity
        hours_yr = 365*24
        ff_om_fix = self.c_ff_fix*self.ff_size
        ff_om_var = self.c_ff_var*kw_plant*hours_yr
        return f * (ff_om_fix+ff_om_var)

    def _ff_total(self, f):
        ff_om = self._ff_om(f)
        ff_replace = self._ff_replace(f)
        return f * ff_om+ff_replace

    # CARBON TAX

    def co2_emit(self, f):
        kw_plant = self.ff_size*self.ff_capacity
        hours_yr = 365*24
        return f * (self.ff_emit/1e3*kw_plant*hours_yr)

    def co2_tax(self, t, f):
        co2_emit = self.co2_emit(f)
        return co2_emit * (self.c_co2_init*((1+self.c_co2_inc)**t))

    # RENEWABLE PLANTS

    def _res_cap(self, v, a):
        return a * (self.c_res_cap[v]*self.res_size)

    def _res_replace(self, v, r):
        return r * (self.c_res_cap[v]*self.res_size/self.res_lifetime)

    def _res_total(self, v, r, a):
        res_cap = self._res_cap(v, a)
        res_replace = self._res_replace(v, r)
        return a * res_cap + r * res_replace

    # BATTERIES

    def _bss_cap(self, v, r, a):
        kwh_bss = self.storage_mix[0] * (self._storage_kwh(r, a) - self._storage_kwh(r, 0))
        return self.c_bss_cap[v]*kwh_bss

    def _bss_om(self, r, a):
        kwh_bss = self.storage_mix[0] * (self._storage_kwh(r, a) - self._storage_kwh(r, 0))
        bss_om_fix = self.c_bss_fix*kwh_bss/(365*24)
        bss_om_var = self.c_bss_var*kwh_bss
        return bss_om_fix+bss_om_var

    def _bss_total(self, v, r, a):
        bss_cap = self._bss_cap(v, r, a)
        bss_om = self._bss_om(r, a)
        return bss_cap+bss_om

    # PUMPED HYDRO

    def _phs_cap(self, v, r, a):
        kwh_phs = self.storage_mix[1] * (self._storage_kwh(r, a) - self._storage_kwh(r, 0))
        return self.c_phs_cap*kwh_phs

    def _phs_om(self, r, a):
        kwh_phs = self.storage_mix[1] * (self._storage_kwh(r, a) - self._storage_kwh(r, 0))
        phs_om_fix = self.c_phs_fix*kwh_phs/(365*24)
        return phs_om_fix

    def _phs_total(self, v, r, a):
        phs_cap = self._phs_cap(v, r, a)
        phs_om = self._phs_om(r, a)
        return phs_cap+phs_om

    # TOTAL STORAGE

    def _storage_kwh(self, r, a):
        kwh_sys_total = self.ff_size*self.ff_capacity * self.n_plants * 24*365
        res_percent = (r+a)*100 / self.n_plants
        # Model grid reliability issues as exponentially increasing storage requirement.
        storage_percent = self.storage_coefs[0] * np.exp(self.storage_coefs[1]*res_percent) + self.storage_coefs[2]
        return storage_percent/100*kwh_sys_total

    def _storage_cap(self, v, r, a):
        return self._bss_cap(v, r, a) + self._phs_cap(v, r, a)

    def _storage_om(self, r, a):
        return self._bss_om(r, a) + self._phs_om(r, a)

    def _storage_total(self, v, r, a):
        return self._bss_total(v, r, a) + self._phs_total(v, r, a)
