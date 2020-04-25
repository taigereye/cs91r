import numpy as np

import mdp.visuals.bar_plot as bp


def cost_breakdown_wrapper(mdp_fh, policy, policy_type, components, t_range, v=None, percent=False):
    t0 = t_range[0]
    tN = t_range[1]
    x = np.arange(t0, tN)
    x_label = "Time (years)"
    y_label = "Cost (USD)"
    if percent:
        y_label = "Cost (%)"
    if "_VS_" in policy_type:
        if v is not None:
            title = "Cost Breakdown in Stage {}: {}".format()
            y_pair_all = np.asarray([cost_breakdown_single_v(mdp_fh, t0, tN, v, pol, components) for pol in policy])
            return bp.plot_single_bar_stacked_double(x, y_pair_all, x_label, y_label, components, title, percent=percent)
        else:
            title = "Cost Breakdown: {}".format(policy_type)
            y_double = np.asarray([cost_breakdown_all_v(mdp_fh, t0, tN, pol, components) for pol in policy])
            y_pair_all_v = np.asarray([list(item) for item in zip(y_double[0], y_double[1])])
            return bp.plot_multiple_bar_stacked_double(x, y_pair_all_v, x_label, y_label, components, title, percent=percent)
    else:
        if v is not None:
            title = "Cost Breakdown in Stage {}: {}".format(v, policy_type)
            y_all = cost_breakdown_single_v(mdp_fh, t0, tN, v, policy, components)
            return bp.plot_single_bar_stacked(x, y_all, x_label, y_label, components, title, percent=percent)
        else:
            title = "Cost Breakdown: {}".format(policy_type)
            y_all_v = cost_breakdown_all_v(mdp_fh, t0, tN, policy, components)
            return bp.plot_multiple_bar_stacked(x, y_all_v, x_label, y_label, components, title, percent=percent)


def cost_breakdown_single_v(mdp_fh, t0, tN, v, policy, components):
    y_all = []
    for c in components:
        y = np.asarray([mdp_fh.mdp_cost.calc_partial_cost(t, v, r, a, c) for (t, v, r, a) in policy[t0:tN]])
        y_all.append(y)
    y_all = np.stack(np.asarray(y_all), axis=0)
    return y_all


def cost_breakdown_all_v(mdp_fh, t0, tN, policy, components):
    y_all_v = []
    for v in np.arange(mdp_fh.n_tech_stages):
        y_all = []
        for c in components:
            y = np.asarray([mdp_fh.mdp_cost.calc_partial_cost(t, v, r, a, c) for (t, v, r, a) in policy[v][t0:tN]])
            y_all.append(y)
        y_all = np.stack(np.asarray(y_all), axis=0)
        y_all_v.append(y_all)
    return np.asarray(y_all_v)


def cost_by_component_wrapper(mdp_fh, policy, policy_type, component, t_range, v=None):
    t0 =t_range[0]
    tN = t_range[1]
    x = np.arange(t0, tN)
    x_label = "Time (years)"
    y_label = "Cost (USD)"
    if "_VS_" in policy_type:
        if v is not None:
            title = "Cost Component {} in Stage {}: {}".format(component, v, policy_type)
            y_pair_all = np.asarray([cost_by_component_single_v(mdp_fh, t0, tN, v, pol, component) for pol in policy])
            return bp.plot_single_bar_double(x, y_pair_all, x_label, y_label, title)
        else:
            title = "Cost Component {}: {}".format(component, policy_type)
            y_double = np.asarray([cost_by_component_all_v(mdp_fh, t0, tN, v, pol, component) for pol in policy])
            y_pair_all_v = np.asarray([list(item) for item in zip(y_double[0], y_double[1])])
            return bp.plot_multiple_bar_double(x, y_pair_all_v, x_label, y_label, title)
    else:
        if v is not None:
            title = "Cost Component {} in Stage {}: {}".format(component, v, policy_type)
            y = cost_by_component_single_v(mdp_fh, t0, tN, v, policy, component)
            return bp.plot_single_bar(x, y, x_label, y_label, title)
        else:
            title = "Cost Component {}: {}".format(component, policy_type)
            y_v = cost_by_component_all_v(mdp_fh, t0, tN, policy, component)
            return bp.plot_multiple_bar(x, y_v, x_label, y_label, title)


def cost_by_component_single_v(mdp_fh, t0, tN, v, policy, component):
    y = np.asarray([mdp_fh.mdp_cost.calc_partial_cost(t, v, r, a, component) for (t, v, r, a) in policy[t0:tN]])
    return y


def cost_by_component_all_v(mdp_fh, t0, tN, policy, component):
    y_v = []
    for v in np.arange(mdp_fh.n_tech_stages):
        y = np.asarray([mdp_fh.mdp_cost.calc_partial_cost(t, v, r, a, component) for (t, v, r, a) in policy[v][t0:tN]])
        y_v.append(y)
    return np.asarray(y_v)


def total_cost_wrapper(mdp_fh, policy, policy_type, t_range, v=None):
    t0 =t_range[0]
    tN = t_range[1]
    x = np.arange(t0, tN)
    x_label = "Time (years)"
    y_label = "Cost (USD)"
    if "_VS_" in policy_type:
        if v is not None:
            title = "Total Cost in Stage {}: {}".format(v, policy_type)
            y_pair_all = np.asarray([total_cost_single_v(mdp_fh, t0, tN, v, pol) for pol in policy])
            return bp.plot_single_bar_double(x, y_pair_all, x_label, y_label, title)
        else:
            title = "Total Cost: {}".format(policy_type)
            y_double = np.asarray([total_cost_all_v(mdp_fh, t0, tN, pol) for pol in policy])
            y_pair_all_v = np.asarray([list(item) for item in zip(y_double[0], y_double[1])])
            return bp.plot_multiple_bar_double(x, y_pair_all_v, x_label, y_label, title)
    else:
        if v is not None:
            title = "Total Cost in Stage {}: {}".format(v, policy_type)
            y = total_cost_single_v(mdp_fh, t0, tN, v, policy)
            return bp.plot_single_bar(x, y, x_label, y_label, title)
        else:
            title = "Total Cost: {}".format(policy_type)
            y_v = total_cost_all_v(mdp_fh, t0, tN, policy)
            return bp.plot_multiple_bar(x, y_v, x_label, y_label, title)


def total_cost_single_v(mdp_fh, t0, tN, v, policy):
    y = np.asarray([mdp_fh.mdp_cost.calc_total_cost(t, v, r, a) for (t, v, r, a) in policy[t0:tN]])
    return y


def total_cost_all_v(mdp_fh, t0, tN, policy):
    y_v = []
    for v in np.arange(mdp_fh.n_tech_stages):
        y = np.asarray([mdp_fh.mdp_cost.calc_total_cost(t, v, r, a) for (t, v, r, a) in policy[v][t0:tN]])
        y_v.append(y)
    return np.asarray(y_v)


# HELPER FUNCTIONS


def get_arb_policy_trajectory(policy, v):
    n_years = len(policy)
    t = 0
    r = 0
    policy_annotated = []
    policy_annotated.append((t, v, r, policy[0]))
    r += policy[0]
    for step in np.arange(1, n_years):
        a = policy[step]
        policy_annotated.append((t, v, r, a))
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
        policy_annotated.append((t, v, r, a))
        t += 1
        r += a
    return policy_annotated
