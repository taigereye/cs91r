from collections import OrderedDict
import itertools as it
import numpy as np
import mdptoolbox as mtb
from pathlib import Path
from scipy.sparse import csr_matrix


class MdpModelV4():
    def __init__(self):
        self.params_to_policy = OrderedDict()
        self.param_names = ['n_years',
                            'n_tech_stages',
                            'n_plants',
                            'n_tax_levels',
                            'ff_size',
                            'ff_capacity',
                            'ff_lifetime',
                            'res_capacity',
                            'res_lifetime',
                            'c_co2_base_levels',
                            'c_co2_inc_levels',
                            'co2_tax_type',
                            'co2_tax_adjust',
                            'co2_tax_cycle',
                            'c_ff_cap',
                            'c_ff_fix',
                            'c_ff_var',
                            'ff_emit',
                            'c_res_cap',
                            'c_res_fix',
                            'storage_mix',
                            'storage_coefs',
                            'bss_hrs',
                            'c_bss_cap',
                            'c_bss_fix',
                            'c_bss_var',
                            'c_phs_cap',
                            'c_phs_fix',
                            'p_adv_tech',
                            'disc_rate',
                            'emit_targets',
                            'target_max_delta']

    # Create params dict from list of param values in order.
    def create_params(self, param_list):
        params = OrderedDict()
        for i in np.arange(len(self.param_names)):
            params[self.param_names[i]] = param_list[i]
        return params

    # Print params, optimal policy, reward matrix of a MDP instance.
    def print_fh(self, mdp_fh):
        assert(mdp_fh is not None)
        mdp_fh.print_params()
        print("\n\n")
        mdp_fh.print_policy()
        print("\n\n")
        mdp_fh.print_rewards()

    # Create MDP instance given params and run once.
    def run_fh(self, params):
        mdp_fh = MdpFiniteHorizonV4(params.copy())
        mdp_fh.initialize()
        mdp_fh.run()
        return mdp_fh


class MdpFiniteHorizonV4():
    def __init__(self, params):
        self.mdp_inst = None
        self.params = params.copy()
        # Cost
        self.mdp_cost = MdpCostCalculatorV4(params)
        # Parameters
        self.n_years = params['n_years']
        self.n_tech_stages = params['n_tech_stages']
        self.n_plants = params['n_plants']
        self.n_tax_levels = params['n_tax_levels']
        self.n_total_levels = len(params['c_co2_base_levels'])
        self.co2_tax_cycle = params['co2_tax_cycle']
        self.p_adv_tech = params['p_adv_tech']
        self.disc_rate = params['disc_rate']
        self.emit_targets = self.mdp_cost.read_targetsfile(params['emit_targets'])
        self.target_max_delta = params['target_max_delta']
        # Constants
        self.scale_down = 9
        self.n_adjustments = 3
        # Dimensions
        self.A = self.n_plants + 1
        self.S = (self.n_years+1) * self.n_tech_stages * (self.n_plants+1) * self.n_adjustments * self.n_tax_levels
        # States
        self.state_to_id = OrderedDict()
        self.id_to_state = OrderedDict()
        # Matrices
        self.transitions = None
        self.rewards = None

    # Create state space, fill transition probabilities matrix, fill rewards matrix.
    def initialize(self):
        print("\nInitializing MDP V4 ...\n")
        self._enumerate_states()
        self._trans_probs_wrapper()
        self._rewards_wrapper()
        print("Setting up MDP FH...\n")
        self.mdp_inst = mtb.mdp.FiniteHorizon(self.transitions,
                                              self.rewards,
                                              1-self.disc_rate,
                                              self.n_years)
        print("Initialization done.\n")

    # Run MDP once.
    def run(self):
        print("Running MDP V4 ...")
        self.mdp_inst.run()
        print("MDP done.\n")

    # Print current params.
    def print_params(self):
        print("PARAMETERS:\n")
        for k, v in self.params.items():
            print("%s: %s" % (k, v))

    # Print optimal policy.
    def print_policy(self):
        assert self.mdp_inst is not None
        print("OPTIMAL POLICY:\n\nState\t     Time")
        self._print_labeled_matrix(self.mdp_inst.policy)

    # Print cost (rewards) matrix of a single cost component.
    def print_partial_costs(self, component):
        components = ["co2_emit",
                      "co2_tax",
                      "ff_total",
                      "ff_replace",
                      "ff_om",
                      "res_total",
                      "res_cap",
                      "res_om",
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

    # Print rewards matrix.
    def print_rewards(self):
        assert self.mdp_inst is not None
        print("REWARDS MATRIX:")
        self._print_labeled_matrix(self.rewards, to_round=True)

    ## STATE SPACE

    # Create mappings of state vectors to unique integer IDs and vice versa.
    def _enumerate_states(self):
        idx = 0
        iter_states = self._get_iter_states()
        for state in iter_states:
            (t, v, r, l, e) = state
            self.state_to_id[state] = idx
            idx += 1
        self.id_to_state = {v: k for k, v in self.state_to_id.items()}

    ## TRANSITION PROBABILITIES

    def _trans_probs_wrapper(self):
        self.transitions = [None] * self.A
        print("Filling transition probabilities...")
        self._fill_trans_probs()
        print("Transition probabilities done.\n")

    # Fill transition probabilities as list of length A of SxS sparse matrices.
    def _fill_trans_probs(self):
        for a in np.arange(self.A):
            action_matrix = csr_matrix((self.S, self.S), dtype=np.float32).toarray()
            iter_states = self._get_iter_states()
            for state in iter_states:
                (t, v, r, l, e), state_curr, idx_curr = self._breakdown_state(state=state)
                # Transition doesn't matter for last year as long as it exists.
                if t == self.n_years:
                    action_matrix[idx_curr][idx_curr] = 1.0
                    continue
                # Build 0 RES plants
                if a == 0:
                    self._fill_single_action_probs(action_matrix, state_curr, 0)
                # Build at least 1 RES plant
                else:
                    # If action invalid always build maximum plants possible.
                    if a > self.n_plants - r:
                        self._fill_single_action_probs(action_matrix, state_curr, self.n_plants-r)
                    else:
                        self._fill_single_action_probs(action_matrix, state_curr, a)
                # assert np.isclose(np.sum(action_matrix[idx_curr]),
                #                   1.0), np.sum(action_matrix[idx_curr])
                self._normalize_trans_row(action_matrix, state_curr, a)

                self.transitions[a] = action_matrix

    # Fill a SxS matrix of transition probabilities for a given action.
    def _fill_single_action_probs(self, action_matrix, state, a):
        (t, v, r, l, e), state_curr, idx_curr = self._breakdown_state(state=state)
        # Next state if tech stage does not advance.
        state_next = self.single_state_transition(state_curr, a)
        idx_next = self.state_to_id[state_next]
        # Have not reached last tech stage so possible to advance.
        if v < self.n_tech_stages - 1:
            # Next state if tech stage does advance.
            state_next_v = self.single_state_transition(state_curr, a, inc_tech_stage=True)
            idx_next_v = self.state_to_id[state_next_v]
            # Tech stage may remain the same.
            action_matrix[idx_curr][idx_next] = 1.0 - self.p_adv_tech[v]
            # Tech stage may advance by 1.
            action_matrix[idx_curr][idx_next_v] = self.p_adv_tech[v]
        else:
            # Tech stage must remain the same.
            action_matrix[idx_curr][idx_next] = 1.0

    ## REWARDS

    def _rewards_wrapper(self):
        self.rewards = np.zeros([self.S, self.A])
        print("Filling rewards...")
        self._fill_rewards()
        print("Rewards done.\n")

    # Fill rewards as SxA matrix.
    def _fill_rewards(self):
        for a in np.arange(self.A):
            for s in np.arange(self.S):
                (t, v, r, l, e), state, idx = self._breakdown_state(idx=s)
                # Sanity check for integer id.
                assert(idx == s)
                cost = self.mdp_cost.calc_total_cost(state, a)
                if cost < np.inf:
                    cost /= self.scale_down
                # Model reward as negative cost.
                self.rewards[idx][a] = -1 * cost

    ## HELPER FUNCTIONS

    def _breakdown_state(self, idx=None, state=None):
        if idx is None:
            (t, v, r, l, e) = state
            idx = self.state_to_id[state]
        if state is None:
            state = self.id_to_state[idx]
            (t, v, r, l, e) = state
        return ((t, v, r, l, e), state, idx)

    def calc_emit_delta(self, t, f):
        target = self.emit_targets[t//self.co2_tax_cycle]
        co2_emit = self.mdp_cost.co2_emit(f)
        return target, co2_emit - target

    def calc_next_adjustment(self, t, r):
        f = self.n_plants - r
        target, delta = self.calc_emit_delta(t, f)
        if abs(delta)/target < self.target_max_delta/100:
            return 0
        elif delta < 0:
            return 1
        elif delta > 0:
            return 2

    def _fill_partial_costs(self, component):
        self._enumerate_states()
        mdp_cost = MdpCostCalculatorV4(self.params)
        # Absolute cost.
        costs = np.zeros([self.S, self.A])
        # Percentage cost.
        percents = np.zeros([self.S, self.A])
        for a in np.arange(self.A):
            for s in np.arange(self.S):
                state = self.id_to_state[s]
                idx = self.state_to_id[state]
                # Sanity check for integer id.
                assert(idx == s)
                costs[idx][a] = mdp_cost.calc_partial_cost(state, a, component)
                # Keep cost positive to save printing space.
                if costs[idx][a] == np.inf:
                    percents[idx][a] = np.inf
                else:
                    costs[idx][a] /= self.scale_down
                    total_cost = mdp_cost.calc_total_cost(state, a)
                    if total_cost == 0:
                        percents[idx][a] = 0
                    else:
                        percents[idx][a] = (costs[idx][a]*100) / total_cost
        return costs, percents

    # Iterate over state space.
    def _get_iter_states(self):
        idx_default = self.n_total_levels//2
        l_lowest = (idx_default - (self.n_tax_levels-1)//2)
        l_highest = idx_default + (self.n_tax_levels-1)//2
        return it.product(np.arange(self.n_years+1),
                          np.arange(self.n_tech_stages),
                          np.arange(self.n_plants+1),
                          np.arange(l_lowest, l_highest+1),
                          np.arange(self.n_adjustments))

    # Normalize a row in a SxS transition probabilities matrix.
    def _normalize_trans_row(self, action_matrix, state_curr, a):
        idx_curr = self.state_to_id[state_curr]
        action_matrix[idx_curr] = action_matrix[idx_curr] / np.sum(action_matrix[idx_curr])

    # Print a Sx__ matrix with rows labeled as state vectors.
    def _print_labeled_matrix(self, matrix, to_round=False, linewidth=300, precision=3):
        np.set_printoptions(linewidth=linewidth, precision=precision, floatmode='maxprec')
        for row, state in zip(matrix, self._get_iter_states()):
            (t, v, r, l, e) = state
            print("({:02d},{:d},{:02d},{:d},{:d}) : ".format(t, v, r, l, e), end="")
            if to_round:
                print(np.array([round(i) for i in row]))
            else:
                print(row)

    # Find next state given whether tech stage advances.
    def single_state_transition(self, state, a, inc_tech_stage=False):
        (t, v, r, l, e) = state
        # Time and RES plants will always update.
        t_updated = t + 1
        r_updated = r + a
        # Tech stage may update (advance).
        if inc_tech_stage:
            v_updated = v + 1
        else:
            v_updated = v
        # Tax level and delta emissions target will update if end of cycle reached.
        l_updated, e_updated = self.update_state_end_of_cycle(state)
        state_next = (t_updated, v_updated, r_updated, l_updated, e_updated)
        return state_next

    # If end of tax cycle reached, update tax level and delta emissions target.
    def update_state_end_of_cycle(self, state):
        (t, v, r, l, e) = state
        l_updated = l
        e_updated = e
        idx_default = self.n_total_levels//2
        if t > 5 and t % self.co2_tax_cycle == 0:
            if e == 0:
                l_updated = l
            elif e == 1:
                l_updated = max(l-1, idx_default-self.n_tax_levels//2)
            elif e == 2:

                l_updated = min(l+1, idx_default+self.n_tax_levels//2)
            e_updated = self.calc_next_adjustment(t, r)
        return l_updated, e_updated


class MdpCostCalculatorV4():
    def __init__(self, params):
        self.params = params.copy()
        # General
        self.n_plants = params['n_plants']
        self.n_tax_levels = params['n_tax_levels']
        self.n_total_levels = len(params['c_co2_base_levels'])
        # CO2 tax
        self.c_co2_base_levels = params['c_co2_base_levels']
        self.c_co2_base = self.c_co2_base_levels[self.n_total_levels//2]
        self.c_co2_inc_levels = params['c_co2_inc_levels']
        self.c_co2_inc = self.c_co2_inc_levels[self.n_total_levels//2]
        self.co2_tax_type = params['co2_tax_type']
        self.co2_tax_adjust = params['co2_tax_adjust']
        self.co2_tax_cycle = params['co2_tax_cycle']
        self.emit_targets = self.read_targetsfile(params['emit_targets'])
        self.target_max_delta = params['target_max_delta']
        # FF plants
        self.ff_size = params['ff_size']
        self.ff_capacity = params['ff_capacity']
        self.ff_lifetime = params['ff_lifetime']
        self.c_ff_cap = params['c_ff_cap']
        self.c_ff_fix = params['c_ff_fix']
        self.c_ff_var = params['c_ff_var']
        self.ff_emit = params['ff_emit']
        # RES plants
        self.res_size = params['ff_size']*params['ff_capacity']/params['res_capacity']
        self.res_capacity = params['res_capacity']
        self.res_lifetime = params['res_lifetime']
        self.c_res_cap = params['c_res_cap']
        self.c_res_fix = params['c_res_fix']
        # Storage
        self.storage_mix = params['storage_mix']
        self.storage_coefs = params['storage_coefs']
        self.bss_hrs = params['bss_hrs']
        self.c_bss_cap = params['c_bss_cap']
        self.c_bss_fix = params['c_bss_fix']
        self.c_bss_var = params['c_bss_var']
        self.c_phs_cap = params['c_phs_cap']
        self.c_phs_fix = params['c_phs_fix']

    # Calculate a single cost component for a given state.
    def calc_partial_cost(self, state, a, component):
        (t, v, r, l, e) = state
        f = self.n_plants-(r+a)
        if f < 0:
            return np.inf
        cost = 0
        if component == "co2_emit":
            cost = self.co2_emit(f)
        elif component == "co2_tax":
            cost = self.co2_tax(t, l, f)
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
        elif component == "res_om":
            cost = self._res_om(v, r, a)
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

    # Calculate total cost for a given state.
    def calc_total_cost(self, state, a):
        (t, v, r, l, e) = state
        if a + r > self.n_plants:
            return np.inf
        f = self.n_plants - (r+a)
        co2_tax = self.co2_tax(t, l, f)
        ff_total = self._ff_total(f)
        res_total = self._res_total(v, r, a)
        storage_total = self._storage_total(v, r, a)
        return co2_tax + ff_total + res_total + storage_total

    ## CARBON TAX

    def co2_emit(self, f):
        kw_plant = self.ff_size*self.ff_capacity
        hours_yr = 365*24
        return f * (self.ff_emit/1e3*kw_plant*hours_yr)

    def co2_price(self, t, l, f):
        c_co2_base, c_co2_inc = self._adjust_co2_tax(l)
        if self.co2_tax_type == "LIN":
            return self._co2_tax_linear(t, f, c_co2_base, c_co2_inc)
        elif self.co2_tax_type == "EXP":
            return self._co2_tax_exponential(t, f, c_co2_base, c_co2_inc)
        else:
            raise ValueError("co2_tax_type must be LIN or EXP: {}".format(self.co2_tax_type))

    def co2_tax(self, t, l, f):
        co2_emit = self.co2_emit(f)
        return co2_emit * self.co2_price(t, l, f)

    def _co2_tax_linear(self, t, f, c_co2_base, c_co2_inc):
        return c_co2_base + (c_co2_inc*t)

    def _co2_tax_exponential(self, t, f, c_co2_base, c_co2_inc):
        return c_co2_base * ((1+c_co2_inc/100)**t)

    def _adjust_co2_tax(self, l):
        # Default CO2 tax is always middle level.
        c_co2_base = self.c_co2_base_levels[len(self.c_co2_base_levels)//2]
        c_co2_inc = self.c_co2_inc_levels[len(self.c_co2_inc_levels)//2]
        if self.n_tax_levels == 1:
            return c_co2_base, c_co2_inc
        elif self.co2_tax_adjust == "BASE":
            c_co2_base = self.c_co2_base_levels[l]
        elif self.co2_tax_adjust == "INC":
            c_co2_inc = self.c_co2_inc_levels[l]
        else:
            raise ValueError("co2_tax_adjust must be BASE or INC: {}".format(self.co2_tax_type))
        return c_co2_base, c_co2_inc

    ## FOSSIL FUEL PLANTS

    def _ff_replace(self, f):
        return f * (self.c_ff_cap*self.ff_size/self.ff_lifetime)

    def _ff_om(self, f):
        kw_plant = self.ff_size*self.ff_capacity
        hours_yr = 365*24
        ff_om_fix = self.c_ff_fix*self.ff_size
        ff_om_var = 10*self.c_ff_var*kw_plant*hours_yr
        return f * (ff_om_fix+ff_om_var)

    def _ff_total(self, f):
        ff_om = self._ff_om(f)
        ff_replace = self._ff_replace(f)
        return ff_om+ff_replace

    ## RENEWABLE PLANTS

    def _res_cap(self, v, a):
        return a * (self.c_res_cap[v]*self.res_size)

    def _res_om(self, v, r, a):
        res_om_fix = self.c_res_fix*self.res_size
        return (r + a) * res_om_fix

    def _res_replace(self, v, r):
        return r * (self.c_res_cap[v]*self.res_size/self.res_lifetime)

    def _res_total(self, v, r, a):
        res_cap = self._res_cap(v, a)
        res_om = self._res_om(v, r, a)
        res_replace = self._res_replace(v, r)
        return res_cap+res_om+res_replace

    ## BATTERIES

    def _bss_cap(self, v, r, a):
        kwh_bss = self.storage_mix[0] * (self._storage_kwh(r, a) - self._storage_kwh(r, 0))
        return self.c_bss_cap[v]*kwh_bss

    def _bss_om(self, r, a):
        kwh_bss = self.storage_mix[0]*self._storage_kwh(r, a)
        bss_om_fix = self.c_bss_fix*kwh_bss/(365*24)
        bss_om_var = self.c_bss_var*kwh_bss
        return bss_om_fix+bss_om_var

    def _bss_total(self, v, r, a):
        bss_cap = self._bss_cap(v, r, a)
        bss_om = self._bss_om(r, a)
        return bss_cap+bss_om

    ## PUMPED HYDRO

    def _phs_cap(self, v, r, a):
        kwh_phs = self.storage_mix[1] * (self._storage_kwh(r, a) - self._storage_kwh(r, 0))
        return self.c_phs_cap*kwh_phs

    def _phs_om(self, r, a):
        kwh_phs = self.storage_mix[1]*self._storage_kwh(r, a)
        phs_om_fix = self.c_phs_fix*kwh_phs/(365*24)
        return phs_om_fix

    def _phs_total(self, v, r, a):
        phs_cap = self._phs_cap(v, r, a)
        phs_om = self._phs_om(r, a)
        return phs_cap+phs_om

    ## ALL STORAGE

    def _storage_kwh(self, r, a):
        kwh_sys_total = self.ff_size*self.ff_capacity * self.n_plants * 24*365
        res_percent = (r+a)*100 / self.n_plants
        storage_percent = self.storage_coefs[0] * np.exp(self.storage_coefs[1]*res_percent) + self.storage_coefs[2]
        return storage_percent/100*kwh_sys_total

    def _storage_cap(self, v, r, a):
        return self._bss_cap(v, r, a) + self._phs_cap(v, r, a)

    def _storage_om(self, r, a):
        return self._bss_om(r, a) + self._phs_om(r, a)

    def _storage_total(self, v, r, a):
        return self._bss_total(v, r, a) + self._phs_total(v, r, a)

    ## HELPER FUNCTIONS

    def read_targetsfile(self, targetsfile):
        targets_dir = Path("visuals/v4/targets")
        tf = targets_dir / "e_v4_{}.txt".format(targetsfile)
        with open(tf, 'r') as targetsfile:
            emit_targets = eval(targetsfile.read())
        targetsfile.close()
        return emit_targets['y']
