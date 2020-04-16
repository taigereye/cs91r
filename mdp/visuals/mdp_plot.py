import matplotlib.pyplot as plt
import matplotlib.colors as mcols
import numpy as np

COLORS = mcols.TABLEAU_COLORS


def label_bars_above(rects, bar_labels, ax):
    for rect, label in zip(rects, bar_labels):
        x = rect.get_x() + rect.get_width()/2.
        y = rect.get_height()
        ax.annotate("{}".format(label), (x, y), xytext=(0, 5),
                    textcoords="offset points", ha='center', va='bottom')


def get_opt_policy_trajectory(mdp_fh, v):
    opt_policy = mdp_fh.mdp_inst.policy
    actions = []
    t = 0
    r = 0
    for step in np.arange(0, mdp_fh.n_years):
        state = (t, v, r)
        idx = mdp_fh.state_to_id[state]
        a = opt_policy[idx][step]
        actions.append((t, v, r, a))
        t += 1
        r += a
    return actions


def plot_multiple_bar(x, y_all, x_label, y_label, title, legend_labels, colors):
    fig, ax = plt.subplots()
    for i in range(len(y_all)):
        y = y_all[i]
        c = colors[i]
        ax.bar(x+(0.1*i), y, width=0.10, color=c)
    ax.set(xlabel=x_label, ylabel=y_label)
    ax.grid()
    ax.legend()
    ax.set_title(title)
    return fig


def plot_single_bar(x, y, x_label, y_label, title, bar_labels=None):
    fig, ax = plt.subplots()
    ax.bar(x, y, width=0.20, color='b')
    if bar_labels:
        label_bars_above(ax.patches, bar_labels, ax)
    ax.set(xlabel=x_label, ylabel=y_label)
    ax.grid()
    ax.set_title(title)
    return fig


def plot_stacked_bar(x, y_all, x_label, y_label, title, legend_labels, colors, percent=False):
    fig, ax = plt.subplots()
    if percent:
        y_total = np.array([sum(y) for y in zip(*y_all)])
        y_all = [y/y_total for y in y_all]
    ax.bar(x, y_all[0], width=0.20, label=legend_labels[0])
    for i in np.arange(1, len(y_all)):
        y = y_all[i]
        label = legend_labels[i]
        ax.bar(x, y, width=0.20, bottom=y_all[i-1], label=label)
    ax.set(xlabel=x_label, ylabel=y_label)
    ax.grid()
    ax.legend()
    ax.set_title(title)
    return fig


def total_cost_by_rplants(mdp_fh, r, v):
    x = np.arange(1, mdp_fh.n_years+1)
    y_rplants = []
    for a in np.arange(mdp_fh.n_plants-r):
        y = np.array([mdp_fh.calc_total_cost(t, v, r, a) for t in x])
        y_rplants.append(y)
    if r > 0:
        for y in y_rplants:
            y[0] = 0
    x_label = "Time (years)"
    y_label = "Cost (USD)"
    title = "Total Cost Given {} RES Plants".format(r)
    legend_labels = [str(i) for i in np.arange(mdp_fh.n_plants-r)]
    return plot_multiple_bar(x, y_rplants, x_label, y_label, title, legend_labels, COLORS)


def cost_component_by_rplants(mdp_fh, r, v, component):
    x = np.arange(1, mdp_fh.n_years+1)
    y_rplants = []
    for a in (mdp_fh.n_plants-r):
        y = np.array([mdp_fh.calc_partial_cost(t, v, r, a, component) for t in x])
        y_rplants.append(y)
    if r > 0:
        for y in y_rplants:
            y[0] = 0
    x_label = "Time (years)"
    y_label = "Cost (USD)"
    title = "Cost Component: {} Given {} RES Plants".format(component, r)
    legend_labels = [str(i) for i in np.arange(mdp_fh.n_plants-r)]
    return plot_multiple_bar(x, y_rplants, x_label, y_label, title, legend_labels, COLORS)


def cost_breakdown(mdp_fh, v, policy, policy_type, percent=False):
    x = np.arange(1, mdp_fh.n_years+1)
    components = ["fplants_OM", "co2_tax", "rplants_cap", "rplants_replace", "storage_cap", "storage_OM"]
    y_components = []
    for component in components:
        y = np.array([mdp_fh.calc_partial_cost(t, v, r, a, component) for (t, v, r, a) in policy])
        y_components.append(y)
    x_label = "Time (years)"
    y_label = "Cost (USD)"
    if percent:
        y_label = "Cost (%)"
    title = "{} Cost Breakdown".format(policy_type)
    return plot_stacked_bar(x, y_components, x_label, y_label, title, components, COLORS, percent)


def cost_by_component(mdp_fh, v, policy, policy_type, component):
    x = np.arange(1, mdp_fh.n_years+1)
    y = np.array([mdp_fh.calc_partial_cost(t, v, r, a, component) for (t, v, r, a) in policy])
    totals = np.array([mdp_fh.calc_total_cost(t, v, r, a) for (t, v, r, a) in policy])
    percents = y*100 / totals
    x_label = "Time (years)"
    y_label = "Cost (USD)"
    title = "{} Cost Component: {}".format(policy_type, component)
    return plot_single_bar(x, y, x_label, y_label, title, percents)


def total_cost(mdp_fh, v, policy, policy_type):
    x = np.arange(1, mdp_fh.n_years+1)
    y = np.array([mdp_fh.calc_total_cost(t, v, r, a) for (t, v, r, a) in policy])
    fig, ax = plt.subplots()
    x_label = "Time (years)"
    y_label = "Cost (USD)"
    title = "{} Total Cost".format(policy_type)
    return plot_single_bar(x, y, x_label, y_label, title)
