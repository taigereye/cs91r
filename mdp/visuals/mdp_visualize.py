import numpy as np

from mdp.models.mdp_v2 import MdpModelV2
import mdp.visuals.mdp_plot as mplt


# COSTS

# Total annual cost breakdown into ff, res, co2, bhs, phs costs with fixed tech stage (single or all)

def cost_breakdown_wrapper(mdp_fh, policy, policy_type, components, t_range, v=None, percent=False):
    t0 = t_range[0]
    tN = t_range[1]
    x = convert_x_time_2020(t0, tN)
    x_label = "Time"
    if "_VS_" in policy_type:
        if v is not None:
            title = "Annual Cost Breakdown in Tech Stage {}: {}".format()
            y_pair_all = np.asarray([cost_breakdown_single_v(mdp_fh, t0, tN, v, pol, components) for pol in policy])
            y_pair_all, scale_str = scale_y_dollar_data(y_pair_all)
            y_label = format_ylabel_dollar(scale_str, is_annual=True)
            if percent:
                y_label = "Cost (%)"
            return mplt.plot_single_bar_stacked_double(x, y_pair_all, x_label, y_label,
                                                       components, title, percent=percent)
        else:
            title = "Annual Cost Breakdown: {}".format(policy_type)
            y_double = np.asarray([cost_breakdown_all_v(mdp_fh, t0, tN, pol, components) for pol in policy])
            y_pair_all_v = np.asarray([list(item) for item in zip(y_double[0], y_double[1])])
            y_pair_all_v, scale_str = scale_y_dollar_data(y_pair_all_v)
            y_label = format_ylabel_dollar(scale_str, is_annual=True)
            if percent:
                y_label = "Cost (%)"
            return mplt.plot_multiple_bar_stacked_double(x, y_pair_all_v, x_label, y_label,
                                                         components, title, percent=percent)
    else:
        if v is not None:
            title = "Annual Cost Breakdown in Tech Stage {}: {}".format(v, policy_type)
            y_all = cost_breakdown_single_v(mdp_fh, t0, tN, v, policy, components)
            y_all, scale_str = scale_y_dollar_data(y_all)
            y_label = format_ylabel_dollar(scale_str, is_annual=True)
            if percent:
                y_label = "Cost (%)"
            components = ["CO2 Tax", "FF", "RES", "BSS", "PHS"]
            return mplt.plot_single_bar_stacked(x, y_all, x_label, y_label,
                                                components, title, percent=percent)
        else:
            title = "Annual Cost Breakdown: {}".format(policy_type)
            y_all_v = cost_breakdown_all_v(mdp_fh, t0, tN, policy, components)
            y_all_v, scale_str = scale_y_dollar_data(y_all_v)
            y_label = format_ylabel_dollar(scale_str, is_annual=True)
            if percent:
                y_label = "Cost (%)"
            return mplt.plot_multiple_bar_stacked(x, y_all_v, x_label, y_label,
                                                  components, title, percent=percent)


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


# Single component of total annual cost with fixed tech stage (single or all)

def cost_by_component_wrapper(mdp_fh, policy, policy_type, component, t_range, v=None):
    t0 = t_range[0]
    tN = t_range[1]
    x = convert_x_time_2020(t0, tN)
    x_label = "Time"
    if "_VS_" in policy_type:
        if v is not None:
            title = "Cost Component {} in Tech Stage {}: {}".format(component, v, policy_type)
            y_pair_all = np.asarray([cost_by_component_single_v(mdp_fh, t0, tN, v, pol, component) for pol in policy])
            y_pair_all, scale_str = scale_y_dollar_data(y_pair_all)
            y_label = format_ylabel_dollar(scale_str, is_annual=True)
            return mplt.plot_single_bar_double(x, y_pair_all, x_label, y_label, title)
        else:
            title = "Cost Component {}: {}".format(component, policy_type)
            y_double = np.asarray([cost_by_component_all_v(mdp_fh, t0, tN, v, pol, component) for pol in policy])
            y_pair_all_v = np.asarray([list(item) for item in zip(y_double[0], y_double[1])])
            y_pair_all_v, scale_str = scale_y_dollar_data(y_pair_all_v)
            y_label = format_ylabel_dollar(scale_str, is_annual=True)
            return mplt.plot_multiple_bar_double(x, y_pair_all_v, x_label, y_label, title)
    else:
        if v is not None:
            title = "Cost Component {} in Tech Stage {}: {}".format(component, v, policy_type)
            y = cost_by_component_single_v(mdp_fh, t0, tN, v, policy, component)
            y, scale_str = scale_y_dollar_data(y)
            y_label = format_ylabel_dollar(scale_str, is_annual=True)
            return mplt.plot_single_bar(x, y, x_label, y_label, title)
        else:
            title = "Cost Component {}: {}".format(component, policy_type)
            y_v = cost_by_component_all_v(mdp_fh, t0, tN, policy, component)
            y_v, scale_str = scale_y_dollar_data(y_v)
            y_label = format_ylabel_dollar(scale_str, is_annual=True)
            return mplt.plot_multiple_bar(x, y_v, x_label, y_label, ["Tech Stage 0", "Tech Stage 1", "Tech Stage 2"], title)


def cost_by_component_single_v(mdp_fh, t0, tN, v, policy, component):
    y = np.asarray([mdp_fh.mdp_cost.calc_partial_cost(t, v, r, a, component) for t, v, r, a in policy[t0:tN]])
    return y


def cost_by_component_all_v(mdp_fh, t0, tN, policy, component):
    y_v = []
    for v in np.arange(mdp_fh.n_tech_stages):
        y = np.asarray([mdp_fh.mdp_cost.calc_partial_cost(t, v, r, a, component) for t, v, r, a in policy[v][t0:tN]])
        y_v.append(y)
    return np.asarray(y_v)


# Total annual cost with fixed tech stage (single or all)

def total_cost_wrapper(mdp_fh, policy, policy_type, t_range, v=None):
    t0 = t_range[0]
    tN = t_range[1]
    x = convert_x_time_2020(t0, tN)
    x_label = "Time"
    y_label = "Cost (USD)"
    if "_VS_" in policy_type:
        if v is not None:
            y_pair_all = np.asarray([total_cost_single_v(mdp_fh, t0, tN, v, pol) for pol in policy])
            y_pair_all, scale_str = scale_y_dollar_data(y_pair_all)
            y_label = format_ylabel_dollar(scale_str, is_annual=True)
            title = "Total Cost in Tech Stage {}: {}".format(v, policy_type)
            return mplt.plot_single_bar_double(x, y_pair_all, x_label, y_label, title)
        else:
            y_double = np.asarray([total_cost_all_v(mdp_fh, t0, tN, pol) for pol in policy])
            y_pair_all_v = np.asarray([list(item) for item in zip(y_double[0], y_double[1])])
            y_pair_all_v, scale_str = scale_y_dollar_data(y_pair_all_v)
            y_label = format_ylabel_dollar(scale_str, is_annual=True)
            title = "Total Cost: {}".format(policy_type)
            return mplt.plot_multiple_bar_double(x, y_pair_all_v, x_label, y_label, title)
    else:
        if v is not None:
            y = total_cost_single_v(mdp_fh, t0, tN, v, policy)
            y, scale_str = scale_y_dollar_data(y)
            y_label = format_ylabel_dollar(scale_str, is_annual=True)
            title = "Total Cost in Tech Stage {}: {}".format(v, policy_type)
            return mplt.plot_single_bar(x, y, x_label, y_label, title)
        else:
            y_v = total_cost_all_v(mdp_fh, t0, tN, policy)
            y_v, scale_str = scale_y_dollar_data(y_v)
            y_label = format_ylabel_dollar(scale_str, is_annual=True)
            title = "Total Cost: {}".format(policy_type)
            return mplt.plot_multiple_bar(x, y_v, x_label, y_label, ["Tech Stage 0", "Tech Stage 1", "Tech Stage 2"], title)


def total_cost_single_v(mdp_fh, t0, tN, v, policy):
    y = np.asarray([mdp_fh.mdp_cost.calc_total_cost(t, v, r, a) for t, v, r, a in policy[t0:tN]])
    return y


def total_cost_all_v(mdp_fh, t0, tN, policy):
    y_v = []
    for v in np.arange(mdp_fh.n_tech_stages):
        y = np.asarray([mdp_fh.mdp_cost.calc_total_cost(t, v, r, a) for t, v, r, a in policy[v][t0:tN]])
        y_v.append(y)
    return np.asarray(y_v)


# Total annual and cumulative costs combined for different parameters with stochastic tech stage

def total_cost_combine(mdp_fh_costs, param_names, t_range, n_iter, p_adv_vary=True):
    t0 = t_range[0]
    tN = t_range[1]
    x = convert_x_time_2020(t0, tN)
    x_label = "Time"
    labels_ann = [pn + " annual" for pn in param_names]
    labels_cum = [pn + " cumulative" for pn in param_names]
    y_all_ann = []
    for mdp_fh in mdp_fh_costs:
        runs, y_cost = total_cost_probabilistic_v(mdp_fh, t0, tN, n_iter, p_adv_vary=p_adv_vary)
        y_all_ann.append(y_cost)
    y_all_ann, scale_str = scale_y_dollar_data(np.asarray(y_all_ann))
    y_label_ann = "Annual " + format_ylabel_dollar(scale_str)
    y_all_cum = [np.cumsum(y_cost) for y_cost in y_all_ann]
    y_all_cum, scale_str = scale_y_dollar_data(np.asarray(y_all_cum))
    y_label_cum = "Cumulative " + format_ylabel_dollar(scale_str, is_annual=False)
    title = "Average Total Cost Comparison"
    return mplt.plot_multiple_bar_twin_multiple_line(x, y_all_cum, y_all_ann, x_label, y_label_cum,
                                                     y_label_ann, labels_cum, labels_ann, title, colors=['orangered', 'b'])


# Total annual and cumulative costs separately for different parameters with stochastic tech stage

def total_cost_ann_cumu(mdp_fh_costs, param_names, t_range, n_iter, is_annual=True, p_adv_vary=True):
    t0 = t_range[0]
    tN = t_range[1]
    x = convert_x_time_2020(t0, tN)
    x_label = "Time"
    if is_annual:
        y_all = []
        for mdp_fh in mdp_fh_costs:
            runs, y_cost = total_cost_probabilistic_v(mdp_fh, t0, tN, n_iter, p_adv_vary=p_adv_vary)
            y_all.append(y_cost)
        y_all, scale_str = scale_y_dollar_data(np.asarray(y_all))
        y_label = format_ylabel_dollar(scale_str)
        title = "Average Total Annual Cost Comparison"
        return mplt.plot_multiple_bar(x, y_all, x_label, y_label, param_names, title, w=0.25)
    else:
        y_all = []
        for mdp_fh in mdp_fh_costs:
            runs, y_cost = total_cost_probabilistic_v(mdp_fh, t0, tN, n_iter, p_adv_vary=p_adv_vary)
            y_all.append(np.cumsum(y_cost))
        y_all, scale_str = scale_y_dollar_data(np.asarray(y_all))
        y_label = format_ylabel_dollar(scale_str, is_annual=False)
        title = "Average Total Cumulative Cost Comparison"
        return mplt.plot_multiple_bar(x, y_all, x_label, y_label, param_names, title, w=0.25)


def total_cost_probabilistic_v(mdp_fh, t0, tN, n_iter, p_adv_vary=True):
    policy_all = []
    runs = run_techstage_transition(mdp_fh, n_iter, p_adv_vary=p_adv_vary)
    for techstages in runs:
        policy_all.append(get_opt_policy_vary_techstage(mdp_fh, techstages))
    y_cost = [calc_total_cost_annotated_policy(mdp_fh, policy) for policy in policy_all]
    y_cost = np.sum(y_cost, axis=0)/n_iter
    runs = np.sum(runs, axis=0)[t0:tN]/n_iter
    return runs, y_cost


# POLICY

# Optimal policy for each tech stage fixed

def opt_policy_rplants_all_v(mdp_fh, policy, policy_type, t_range, code):
    t0 = t_range[0]
    tN = t_range[1]
    x = convert_x_time_2020(t0, tN)
    x_label = "Time"
    y_label = "Tech Stage"
    policy_v = [extract_idx_annotated_policy(pol[t0:tN], code) for pol in policy]
    y_v = np.asarray(policy_v)
    if code == 'a':
        title = "Newly Built RES Plants: {}".format(policy_type)
    elif code == 'r':
        title = "Total RES Plants: {}".format(policy_type)
    return mplt.plot_heatmap(x, y_v, mdp_fh.n_plants,x_label, y_label, title)


# Average optimal policy with stochastic tech stage

def opt_policy_probabilistic_v(mdp_fh, t_range, n_iter, p_adv_vary=True):
    t0 = t_range[0]
    tN = t_range[1]
    x = convert_x_time_2020(t0, tN)
    x_label = "Time"
    y_label_bars = "Number of Renewable Plants"
    y_label_line = "Average Tech Stage"
    labels = ["New RES Plants", "Total RES Plants"]
    runs, y_a, y_r = avg_a_r_probabilistic_v(mdp_fh, t0, tN, n_iter, p_adv_vary=p_adv_vary)
    title = "Average Optimal Policy with Probabilistic Tech Stage"
    return mplt.plot_single_bar_double_twin_line(x, [y_a, y_r], runs, x_label, y_label_bars,
                                                 y_label_line, labels, title)


def avg_a_r_probabilistic_v(mdp_fh, t0, tN, n_iter, p_adv_vary):
    policy_all = []
    runs = run_techstage_transition(mdp_fh, n_iter, p_adv_vary=p_adv_vary)
    for techstages in runs:
        policy_all.append(get_opt_policy_vary_techstage(mdp_fh, techstages))
    y_a = [extract_idx_annotated_policy(policy[t0:tN], 'a') for policy in policy_all]
    y_a = np.sum(y_a, axis=0)/n_iter
    y_r = [extract_idx_annotated_policy(policy[t0:tN], 'r') for policy in policy_all]
    y_r = np.sum(y_r, axis=0)/n_iter
    runs = np.sum(runs, axis=0)[t0:tN]/n_iter
    return runs, y_a, y_r


# RES PENETRATION

# Average RES penetration for different parameters with stochastic tech stage

def opt_policy_res_percent(mdp_fh_res_percents, t_range, n_iter, params_names, RES=None, p_adv_vary=True):
    t0 = t_range[0]
    tN = t_range[1]
    x = convert_x_time_2020(t0, tN)
    x_label = "Time"
    y_all = res_probabilistic_v(mdp_fh_res_percents, t0, tN, n_iter, p_adv_vary)
    y_all, scale_str = scale_y_dollar_data(y_all)
    y_label = "RES Penetration (%)"
    title = "Average Renewable Penetration"
    return mplt.plot_multiple_line(x, y_all, 1.0, x_label, y_label, params_names, title, RES, is_fixed=False)


def res_probabilistic_v(mdp_fh_res_percents, t0, tN, n_iter, p_adv_vary):
    res_all = []
    for mdp_fh in mdp_fh_res_percents:
        policy_all = []
        runs = run_techstage_transition(mdp_fh, n_iter, p_adv_vary=p_adv_vary)
        for iteration in runs:
            policy_all.append(get_opt_policy_vary_techstage(mdp_fh, iteration))
        y_res = [extract_idx_annotated_policy(policy[t0:tN], 'r') for policy in policy_all]
        y_res = np.sum(y_res, axis=0)/n_iter
        y_res = y_res/mdp_fh.n_plants
        res_all.append(y_res)
    return np.asarray(res_all)


# CO2 EMISSIONS

# Average annual CO2 emissions of optimal policy for different parameters with stochastic tech stage

def opt_policy_co2_emit(mdp_fh_co2_taxes, t_range, n_iter, params_names, CO2=None, p_adv_vary=True):
    t0 = t_range[0]
    tN = t_range[1]
    x = convert_x_time_2020(t0, tN)
    x_label = "Time"
    y_all = co2_emit_probilistic_v(mdp_fh_co2_taxes, t0, tN, n_iter, p_adv_vary)
    y_all, scale_str = scale_y_dollar_data(y_all)
    y_label = format_ylabel_dollar(scale_str).replace("Cost", "CO2 Emissions").replace("USD", "tons")
    title = "Average Annual CO2 Emissions"
    return mplt.plot_multiple_line(x, y_all, 0, x_label, y_label, params_names, title, CO2, colors=['orangered', 'b'], is_fixed=False)


def co2_emit_probilistic_v(mdp_fh_co2_taxes, t0, tN, n_iter, p_adv_vary):
    emit_all = []
    for mdp_fh in mdp_fh_co2_taxes:
        policy_all = []
        runs = run_techstage_transition(mdp_fh, n_iter, p_adv_vary=p_adv_vary)
        for iteration in runs:
            policy_all.append(get_opt_policy_vary_techstage(mdp_fh, iteration))
        y_emit = [calc_co2_emit_annotated_policy(mdp_fh, policy[t0:tN]) for policy in policy_all]
        y_emit = np.sum(y_emit, axis=0)/n_iter
        y_emit = np.cumsum(y_emit)
        emit_all.append(y_emit)
    return np.asarray(emit_all)


# Average annual and cumulative CO2 emissions and taxes of optimal policy with stochastic tech stage

def co2_emit_tax_wrapper(mdp_fh, policy_type, t_range, n_iter, is_annual=False, p_adv_vary=True):
    t0 = t_range[0]
    tN = t_range[1]
    x = convert_x_time_2020(t0, tN)
    x_label = "Time"
    runs, y_r, y_emit, y_tax = avg_co2_probabilistic_v(mdp_fh, t0, tN, n_iter, p_adv_vary=p_adv_vary)
    y_label_r = "Number of Total RES Plants"
    if is_annual:
        y_emit, scale_str = scale_y_dollar_data(y_emit)
        y_label_emit = format_ylabel_dollar(scale_str).replace("Cost", "CO2 Emissions").replace("USD", "tons/yr")
        y_tax, scale_str = scale_y_dollar_data(y_tax)
        y_label_tax = format_ylabel_dollar(scale_str).replace("Cost", "CO2 Tax").replace("USD", "USD/yr")
        labels = ["Total RES Plants", "Annual CO2 Emissions", "Annual CO2 Tax"]
        title = "Annual CO2 Impacts for Average Optimal Policy"
        return mplt.plot_multiple_line_twin_single_bar(x, [y_emit, y_tax], y_r, mdp_fh.n_plants, x_label, [y_label_emit, y_label_tax],
                                                       y_label_r, labels, title, is_annual=is_annual)
    else:
        y_emit_cum = np.cumsum(y_emit)
        y_emit_cum, scale_str = scale_y_dollar_data(y_emit_cum)
        y_label_emit = format_ylabel_dollar(scale_str).replace("Cost", "CO2 Emissions").replace("USD", "tons")
        y_tax_cum = np.cumsum(y_tax)
        y_tax_cum, scale_str = scale_y_dollar_data(y_tax_cum)
        y_label_tax = format_ylabel_dollar(scale_str).replace("Cost", "CO2 Tax")
        labels = ["Total RES Plants", "Cumulative CO2 Emissions", "Cumulative CO2 Tax"]
        title = "Cumulative CO2 Impacts for Average Optimal Policy"
        return mplt.plot_multiple_line_twin_single_bar(x, [y_emit_cum, y_tax_cum], y_r, mdp_fh.n_plants, x_label, [y_label_emit, y_label_tax],
                                                       y_label_r, labels, title, is_annual=is_annual)


def avg_co2_probabilistic_v(mdp_fh, t0, tN, n_iter, p_adv_vary):
    policy_all = []
    runs = run_techstage_transition(mdp_fh, n_iter, p_adv_vary=p_adv_vary)
    for iteration in runs:
        policy_all.append(get_opt_policy_vary_techstage(mdp_fh, iteration))
    y_r = [extract_idx_annotated_policy(policy[t0:tN], 'r') for policy in policy_all]
    y_r = np.sum(y_r, axis=0)/n_iter
    y_emit = [calc_co2_emit_annotated_policy(mdp_fh, policy[t0:tN]) for policy in policy_all]
    y_emit = np.sum(y_emit, axis=0)/n_iter
    y_tax = [calc_co2_tax_annotated_policy(mdp_fh, policy[t0:tN]) for policy in policy_all]
    y_tax = np.sum(y_tax, axis=0)/n_iter
    runs = np.sum(runs, axis=0)[t0:tN]/n_iter
    return runs, y_r, y_emit, y_tax


def calc_co2_emit_annotated_policy(mdp_fh, policy):
    return [mdp_fh.mdp_cost.co2_emit(mdp_fh.n_plants-(r+a)) for t, v, r, a in policy]


def calc_co2_tax_annotated_policy(mdp_fh, policy):
    return [mdp_fh.mdp_cost.co2_tax(t, mdp_fh.n_plants-(r+a)) for t, v, r, a in policy]


# STORAGE


def storage_reductions_wrapper(mdp_fh_reduced, t_range, n_iter, reductions, budget=0, RES=0, p_adv_vary=True):
    t0 = t_range[0]
    tN = t_range[1]
    x = convert_x_time_2020(t0, tN)
    x_label = "Time"
    storage_costs = mdp_extract_storage_costs(mdp_fh_reduced)
    percent_reductions = ["{:.0f}%".format(100-round(frac*100)) for frac in reductions]
    total_s, cum_s = total_cost_reductions_wrapper(mdp_fh_reduced, t0, tN, n_iter, x, x_label, reductions,
                                                   percent_reductions, budget, p_adv_vary)
    total_r = res_penetration_reductions_wrapper(mdp_fh_reduced, t0, tN, n_iter, x, x_label, reductions,
                                                 percent_reductions, RES, p_adv_vary)
    return ((total_s, cum_s), total_r)


# Total annual and total cumulative cost of optimal policy given reductions in bhs/phs costs with stochastic tech stage

def total_cost_reductions_wrapper(mdp_fh_reduced, t0, tN, n_iter, x, x_label, reductions, percent_reductions, budget, p_adv_vary):
    title = "Effect of Storage Costs on Total Annual Cost"
    title_cum = "Effect of Storage Costs on Total Cumulative Cost"
    y_all = total_cost_storage_reductions(mdp_fh_reduced, t0, tN, n_iter, reductions, p_adv_vary)
    y_all_cum = np.cumsum(y_all, axis=1)
    y_all, scale_str = scale_y_dollar_data(y_all)
    y_label = format_ylabel_dollar(scale_str)
    y_all_cum, scale_str = scale_y_dollar_data(y_all_cum)
    y_label_cum = format_ylabel_dollar(scale_str, is_annual=False)
    if budget > 0:
        if scale_str == "thousand":
            budget /= 1e3
        elif scale_str == "million":
            budget /= 1e6
        elif scale_str == "billion":
            budget /= 1e9
        elif scale_str == "trillion":
            budget /= 1e12
    total = mplt.plot_multiple_line(x, y_all, 0, x_label, y_label, percent_reductions,
                                    title, budget, scalar_name="Annual Budget", is_fixed=True)
    cum = mplt.plot_multiple_line(x, y_all_cum, 0, x_label, y_label_cum, percent_reductions,
                                  title_cum, budget, scalar_name="Cumulative Budget", is_fixed=False)
    return total, cum


def total_cost_storage_reductions(mdp_fh_reduced, t0, tN, n_iter, reductions, p_adv_vary):
    storage_all = []
    for mdp_fh in mdp_fh_reduced:
        policy_all = []
        runs = run_techstage_transition(mdp_fh, n_iter, p_adv_vary=p_adv_vary)
        for iteration in runs:
            policy_all.append(get_opt_policy_vary_techstage(mdp_fh, iteration))
        y_cost = [calc_co2_tax_annotated_policy(mdp_fh, policy[t0:tN]) for policy in policy_all]
        y_cost = np.sum(y_cost, axis=0)/n_iter
        storage_all.append(y_cost)
    return np.asarray(storage_all)


# Renewable penetration of optimal policy given reductions in bhs/phs costs with stochastic tech stage

def res_penetration_reductions_wrapper(mdp_fh_reduced, t0, tN, n_iter, x, x_label, reductions, percent_reductions, RES, p_adv_vary):
    y_label = "RES Penetration (%)"
    title = "Effect of Storage Costs on RES Penetration"
    y_all = total_RES_storage_reductions(mdp_fh_reduced, t0, tN, n_iter, reductions, p_adv_vary)
    total = mplt.plot_multiple_line(x, y_all, 100, x_label, y_label, percent_reductions, title,
                                    scalar=RES, scalar_name="Target RES Penetration", is_fixed=True)
    return total


def total_RES_storage_reductions(mdp_fh_reduced, t0, tN, n_iter, reductions, p_adv_vary):
    res_all = []
    for mdp_fh in mdp_fh_reduced:
        policy_all = []
        runs = run_techstage_transition(mdp_fh, n_iter, p_adv_vary=p_adv_vary)
        for iteration in runs:
            policy_all.append(get_opt_policy_vary_techstage(mdp_fh, iteration))
        y_r = [extract_idx_annotated_policy(policy[t0:tN], 'r') for policy in policy_all]
        y_r = np.sum(y_r, axis=0)/n_iter
        y_r_percent = [val*100/mdp_fh.n_plants for val in y_r]
        res_all.append(y_r_percent)
    return np.asarray(res_all)


# HELPER FUNCTIONS


def calculate_cost_scale(min_value):
    min_int = round(min_value)
    scale_exact = len(str(min_int)) - 2
    scale_dollar = min(3 * (scale_exact//3), 12)
    scale_str = None
    if scale_dollar == 3:
        scale_str = "thousand"
    elif scale_dollar == 6:
        scale_str = "million"
    elif scale_dollar == 9:
        scale_str = "billion"
    elif scale_dollar >= 12:
        scale_str = "trillion"
    return 10**scale_dollar, scale_str


def calc_total_cost_annotated_policy(mdp_fh, policy):
    return [mdp_fh.mdp_cost.calc_total_cost(t, int(v), r, a) for t, v, r, a in policy]


def convert_x_time_2020(t0, tN):
    return np.arange(t0+2020, tN+2020)


def extract_idx_annotated_policy (policy, code):
    idx = 0
    if code == 'a':
        idx = 3
    elif code == 'r':
        idx = 2
    policy_extracted = [state[idx] for state in policy]
    return policy_extracted


def format_param_names(param_file):
    pf = param_file
    pf = pf.replace("co2_tax_", "")
    if pf[0].islower():
        if pf == "sternR":
            return "Stern Review"
        elif pf == "africaS":
            return "South Africa"
        else:
            return pf.title()
    else:
        return pf


def format_ylabel_dollar(scale_str, is_annual=True):
    if scale_str:
        if is_annual:
            return "Cost ({} USD/yr)".format(scale_str)
        else:
            return "Cost ({} USD)".format(scale_str)
    else:
        if is_annual:
            return "Cost (USD/yr)"
        else:
            return "Cost (USD)"


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


def get_opt_policy_vary_techstage(mdp_fh, iteration):
    opt_policy = mdp_fh.mdp_inst.policy
    policy_annotated = []
    t = 0
    r = 0
    v = 0
    for step in np.arange(0, mdp_fh.n_years):
        v = iteration[step]
        state = (t, v, r)
        idx = mdp_fh.state_to_id[state]
        a = opt_policy[idx][step]
        policy_annotated.append([t, v, r, a])
        t += 1
        r += a
    return policy_annotated


def mdp_extract_storage_costs(mdp_fh_all):
    storage_costs = []
    for mdp_fh in mdp_fh_all:
        storage_costs.append([mdp_fh.params['c_bss_cap'], mdp_fh.params['c_phs_cap']])
    return storage_costs


def reduce_storage_costs_params(params, frac):
    params_reduced = params.copy()
    params_reduced['c_bss_cap'] = [c*frac for c in params_reduced['c_bss_cap']]
    params_reduced['c_phs_cap'] *= frac
    return params_reduced


def run_techstage_transition(mdp_fh, n_iter, p_adv_vary=True):
    runs = np.zeros([n_iter, mdp_fh.n_years])
    for i in np.arange(n_iter):
        techstage = 0
        if p_adv_vary:
            p_adv = mdp_fh.p_adv_tech[0]
        else:
            p_adv = mdp_fh.p_adv_tech
        for step in np.arange(1, mdp_fh.n_years):
            # Decide whether or not the tech stage advances this year.
            adv = np.random.binomial(1, p_adv)
            if adv and techstage < mdp_fh.n_tech_stages - 1:
                if p_adv_vary:
                    if techstage < mdp_fh.n_tech_stages - 2:
                        p_adv = mdp_fh.p_adv_tech[techstage+1]
                techstage += 1
            runs[i][step] = techstage
    return runs


def scale_y_dollar_data(y):
    min_value = np.min(y)
    if min_value == 0:
        min_value = np.max(y)/2
    scale, scale_str = calculate_cost_scale(min_value)
    y_scaled = y/scale
    return y_scaled, scale_str
