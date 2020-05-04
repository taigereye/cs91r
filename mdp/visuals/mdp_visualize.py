import numpy as np

from mdp.models.mdp_v2 import MdpModelV2
import mdp.visuals.bar_plot as bp


# COSTS

def cost_breakdown_wrapper(mdp_fh, policy, policy_type, components, t_range, v=None, percent=False):
    t0 = t_range[0]
    tN = t_range[1]
    x = np.arange(t0, tN)
    x_label = "Time (Years)"
    y_label = "Cost (USD)"
    if percent:
        y_label = "Cost (%)"
    if "_VS_" in policy_type:
        if v is not None:
            title = "Cost Breakdown in Tech Stage {}: {}".format()
            y_pair_all = np.asarray([cost_breakdown_single_v(mdp_fh, t0, tN, v, pol, components) for pol in policy])
            return bp.plot_single_bar_stacked_double(x, y_pair_all, x_label, y_label, components, title, percent=percent)
        else:
            title = "Cost Breakdown: {}".format(policy_type)
            y_double = np.asarray([cost_breakdown_all_v(mdp_fh, t0, tN, pol, components) for pol in policy])
            y_pair_all_v = np.asarray([list(item) for item in zip(y_double[0], y_double[1])])
            return bp.plot_multiple_bar_stacked_double(x, y_pair_all_v, x_label, y_label, components, title, percent=percent)
    else:
        if v is not None:
            title = "Cost Breakdown in Tech Stage {}: {}".format(v, policy_type)
            y_all = cost_breakdown_single_v(mdp_fh, t0, tN, v, policy, components)
            return bp.plot_single_bar_stacked(x, y_all, x_label, y_label, components, title, percent=percent)
        else:
            title = "Cost Breakdown: {}".format(policy_type)
            y_all_v = cost_breakdown_all_v(mdp_fh, t0, tN, policy, components)
            return bp.plot_multiple_bar_stacked(x, y_all_v, x_label, y_label, components, title, percent=percent)


def cost_breakdown_single_v(mdp_fh, t0, tN, v, policy, components):
    y_all = []
    for c in components:
        y = np.asarray([mdp_fh.mdp_cost.calc_partial_cost(t, v, r, a, c) for t, v, r, a in policy[t0:tN]])
        y_all.append(y)
    y_all = np.stack(np.asarray(y_all), axis=0)
    return y_all


def cost_breakdown_all_v(mdp_fh, t0, tN, policy, components):
    y_all_v = []
    for v in np.arange(mdp_fh.n_tech_stages):
        y_all = []
        for c in components:
            y = np.asarray([mdp_fh.mdp_cost.calc_partial_cost(t, v, r, a, c) for t, v, r, a in policy[v][t0:tN]])
            y_all.append(y)
        y_all = np.stack(np.asarray(y_all), axis=0)
        y_all_v.append(y_all)
    return np.asarray(y_all_v)


def cost_by_component_wrapper(mdp_fh, policy, policy_type, component, t_range, v=None):
    t0 = t_range[0]
    tN = t_range[1]
    x = np.arange(t0, tN)
    x_label = "Time (Years)"
    y_label = "Cost (USD)"
    if "_VS_" in policy_type:
        if v is not None:
            title = "Cost Component {} in Tech Stage {}: {}".format(component, v, policy_type)
            y_pair_all = np.asarray([cost_by_component_single_v(mdp_fh, t0, tN, v, pol, component) for pol in policy])
            return bp.plot_single_bar_double(x, y_pair_all, x_label, y_label, title)
        else:
            title = "Cost Component {}: {}".format(component, policy_type)
            y_double = np.asarray([cost_by_component_all_v(mdp_fh, t0, tN, v, pol, component) for pol in policy])
            y_pair_all_v = np.asarray([list(item) for item in zip(y_double[0], y_double[1])])
            return bp.plot_multiple_bar_double(x, y_pair_all_v, x_label, y_label, title)
    else:
        if v is not None:
            title = "Cost Component {} in Tech Stage {}: {}".format(component, v, policy_type)
            y = cost_by_component_single_v(mdp_fh, t0, tN, v, policy, component)
            return bp.plot_single_bar(x, y, x_label, y_label, title)
        else:
            title = "Cost Component {}: {}".format(component, policy_type)
            y_v = cost_by_component_all_v(mdp_fh, t0, tN, policy, component)
            return bp.plot_multiple_bar(x, y_v, x_label, y_label, title)


def cost_by_component_single_v(mdp_fh, t0, tN, v, policy, component):
    y = np.asarray([mdp_fh.mdp_cost.calc_partial_cost(t, v, r, a, component) for t, v, r, a in policy[t0:tN]])
    return y


def cost_by_component_all_v(mdp_fh, t0, tN, policy, component):
    y_v = []
    for v in np.arange(mdp_fh.n_tech_stages):
        y = np.asarray([mdp_fh.mdp_cost.calc_partial_cost(t, v, r, a, component) for t, v, r, a in policy[v][t0:tN]])
        y_v.append(y)
    return np.asarray(y_v)


def total_cost_wrapper(mdp_fh, policy, policy_type, t_range, v=None):
    t0 = t_range[0]
    tN = t_range[1]
    x = np.arange(t0, tN)
    x_label = "Time (Years)"
    y_label = "Cost (USD)"
    if "_VS_" in policy_type:
        if v is not None:
            title = "Total Cost in Tech Stage {}: {}".format(v, policy_type)
            y_pair_all = np.asarray([total_cost_single_v(mdp_fh, t0, tN, v, pol) for pol in policy])
            return bp.plot_single_bar_double(x, y_pair_all, x_label, y_label, title)
        else:
            title = "Total Cost: {}".format(policy_type)
            y_double = np.asarray([total_cost_all_v(mdp_fh, t0, tN, pol) for pol in policy])
            y_pair_all_v = np.asarray([list(item) for item in zip(y_double[0], y_double[1])])
            return bp.plot_multiple_bar_double(x, y_pair_all_v, x_label, y_label, title)
    else:
        if v is not None:
            title = "Total Cost in Tech Stage {}: {}".format(v, policy_type)
            y = total_cost_single_v(mdp_fh, t0, tN, v, policy)
            return bp.plot_single_bar(x, y, x_label, y_label, title)
        else:
            title = "Total Cost: {}".format(policy_type)
            y_v = total_cost_all_v(mdp_fh, t0, tN, policy)
            return bp.plot_multiple_bar(x, y_v, x_label, y_label, title)


def total_cost_single_v(mdp_fh, t0, tN, v, policy):
    y = np.asarray([mdp_fh.mdp_cost.calc_total_cost(t, v, r, a) for t, v, r, a in policy[t0:tN]])
    return y


def total_cost_all_v(mdp_fh, t0, tN, policy):
    y_v = []
    for v in np.arange(mdp_fh.n_tech_stages):
        y = np.asarray([mdp_fh.mdp_cost.calc_total_cost(t, v, r, a) for t, v, r, a in policy[v][t0:tN]])
        y_v.append(y)
    return np.asarray(y_v)


# POLICY


def policy_plants_all_v(mdp_fh, policy, policy_type, t_range, code):
    t0 = t_range[0]
    tN = t_range[1]
    x = np.arange(t0, tN)
    x_label = "Time (Years)"
    y_label = "Tech Stage"
    policy_v = [extract_idx_annotated_policy(pol[t0:tN], code) for pol in policy]
    y_v = np.asarray(policy_v)
    if code == 'a':
        title = "Renewable Plants Built: {}".format(policy_type)
    elif code == 'r':
        title = "Cumulative Renewable Plants: {}".format(policy_type)
    return bp.plot_heatmap(x, y_v, x_label, y_label, title)


def policy_plants_probabilistic_v(mdp_fh, policy_type, t_range, n_iter):
    t0 = t_range[0]
    tN = t_range[1]
    x = np.arange(t0, tN)
    x_label = "Time (Years)"
    y_bar_label = "Renewable Plants Built"
    y_line_label = "Avg Tech Stage"
    runs, y_a, y_r = avg_policy_probabilistic_v(mdp_fh, t0, tN, n_iter)
    title = "Avg Optimal Policy with Probabilistic Tech Stage"
    return bp.plot_single_bar_double_with_line(x, [y_a, y_r], runs, x_label, y_bar_label, y_line_label, title)


def avg_policy_probabilistic_v(mdp_fh, t0, tN, n_iter):
    policy_all = []
    runs = run_techstage_transition(mdp_fh, n_iter)
    for techstages in runs:
        policy_all.append(get_opt_policy_vary_techstage(mdp_fh, techstages))
    y_a = [extract_idx_annotated_policy(policy[t0:tN], 'a') for policy in policy_all]
    y_a = np.sum(y_a, axis=0)/n_iter
    y_r = [extract_idx_annotated_policy(policy[t0:tN], 'r') for policy in policy_all]
    y_r = np.sum(y_r, axis=0)/n_iter
    runs = np.sum(runs, axis=0)[t0:tN]/n_iter
    return runs, y_a, y_r


# STORAGE


def storage_reductions_wrapper(mdp_fh_reduced, t_range, reductions, budget=None, RESpenetration=None):
    t0 = t_range[0]
    tN = t_range[1]
    x = np.arange(t0, tN)
    x_label = "Time (Years)"
    percent_reductions = ["{:.0f}%".format(frac*100) for frac in reductions]
    if budget:
        y_label = "Cost (USD)"
        y_all = total_cost_storage_reductions(mdp_fh_reduced, t0, tN, reductions)
        title = "Effect of Storage Costs on Total Cost"
        return bp.plot_multiple_line(x, y_all, x_label, y_label, percent_reductions, title, scalar=budget)
    elif RESpenetration:
        y_label = "RES Penetration (%)"
        y_all = total_RES_storage_reductions(mdp_fh_reduced, t0, tN, reductions)
        title = "Effect of Storage Costs on RES Penetration"
        return bp.plot_multiple_line(x, y_all, x_label, y_label, percent_reductions, title, scalar=RESpenetration)


def total_cost_storage_reductions(mdp_fh_reduced, t0, tN, reductions):
    storage_all = []
    for mdp_fh in mdp_fh_reduced:
        # Make all cost reductions relative to tech stage 0.
        opt_policy = get_opt_policy_trajectory(mdp_fh, 0)
        y = total_cost_single_v(mdp_fh, t0, tN, 0, opt_policy)
        storage_all.append(y)
    return np.asarray(storage_all)


def total_RES_storage_reductions(mdp_fh_reduced, t0, tN, reductions):
    storage_all = []
    for mdp_fh in mdp_fh_reduced:
        # Make all cost reductions relative to tech stage 0.
        opt_policy = get_opt_policy_trajectory(mdp_fh, 0)
        y_r = extract_idx_annotated_policy(opt_policy[t0:tN], 'r')
        y = [val*100/mdp_fh.n_plants for val in y_r]
        storage_all.append(y)
    return np.asarray(storage_all)


# HELPER FUNCTIONS


def extract_idx_annotated_policy (policy, code):
    idx = 0
    if code == 'a':
        idx = 3
    elif code == 'r':
        idx = 2
    policy_extracted = [state[idx] for state in policy]
    return policy_extracted


def get_arb_policy_trajectory(policy, v):
    n_years = len(policy)
    t = 0
    r = 0
    policy_annotated = []
    policy_annotated.append([t, v, r, policy[0]])
    r += policy[0]
    for step in np.arange(1, n_years):
        a = policy[step]
        policy_annotated.append([t, v, r, a])
        t += 1
        r += a
    return policy_annotated


def get_opt_policy_trajectory(mdp_fh, v):
    opt_policy = mdp_fh.mdp_inst.policy
    policy_annotated = []
    t = 0
    r = 0
    for step in np.arange(0, mdp_fh.n_years):
        state = (t, v, r)
        idx = mdp_fh.state_to_id[state]
        a = opt_policy[idx][step]
        policy_annotated.append([t, v, r, a])
        t += 1
        r += a
    return policy_annotated


def get_opt_policy_vary_techstage(mdp_fh, techstages):
    opt_policy = mdp_fh.mdp_inst.policy
    policy_annotated = []
    t = 0
    r = 0
    v = 0
    for step in np.arange(0, mdp_fh.n_years):
        v = techstages[step]
        state = (t, v, r)
        idx = mdp_fh.state_to_id[state]
        a = opt_policy[idx][step]
        policy_annotated.append([t, v, r, a])
        t += 1
        r += a
    return policy_annotated


def reduce_storage_costs_params(params, frac):
    params_reduced = params
    params_reduced['c_bss_cap'] = [c*frac for c in params_reduced['c_bss_cap']]
    params_reduced['c_phs_cap'] *= frac
    return params_reduced


def run_techstage_transition(mdp_fh, n_iter):
    runs = np.zeros([n_iter, mdp_fh.n_years])
    for i in np.arange(n_iter):
        techstage = 0
        for step in np.arange(1, mdp_fh.n_years):
            # Whether or not the tech stage advances this year.
            adv = np.random.binomial(1, mdp_fh.p_adv_tech_stage)
            if adv and techstage < 2:
                techstage += 1
            runs[i][step] = techstage
    return runs
