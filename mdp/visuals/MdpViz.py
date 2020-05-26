from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MdpDataGatherer():
    def __init__(self, mdp_model, n_iter, t_range):
        self.mdp_model = mdp_model
        self.instances = OrderedDict()
        self.n_iter = n_iter
        self.t0 = t_range[0]
        self.tN = t_range[1]

    def add_mdp_instance(self, paramsfile, params):
        assert(self.mdp_model.param_names == list(params.keys()))
        mdp_fh = self.mdp_model.run_fh(params)
        self.instances[paramsfile] = mdp_fh
        return mdp_fh

    ## COST

    # Get all cost components averaged across stochastic tech stage.
    def cost_breakdown_components(self, mdp_fh, components, is_percent=False):
        policy_all, avg_techstages = self.aggregate_annotated_policies(mdp_fh)
        y_all = []
        for policy in policy_all:
            y_components = []
            for c in components:
                y = [mdp_fh.mdp_cost.calc_partial_cost(state, a, c) for state, a in policy]
                y_components.append(y)
            y_components = np.stack(np.asarray(y_components), axis=0)
            y_all.append(y_components)
        y_mean = np.sum(y_all, axis=0)/self.n_iter
        if is_percent:
            y_mean = y_mean / np.sum(y_mean, axis=0)
        return y_mean

    # Get single cost component averaged across stochastic tech stage.
    def cost_single_component(self, mdp_fh, component):
        policy_all, avg_techstages = self.aggregate_annotated_policies(mdp_fh)
        y_all = []
        for policy in policy_all:
            y_component = [mdp_fh.mdp_cost.calc_partial_cost(state, a, component) for state, a in policy]
            y_all.append(y_component)
        y_mean, y_lower, y_upper = self.calc_data_bounds_std(y_all)
        return (y_mean, y_lower, y_upper)

    # Get total cost averaged across stochastic tech stage.
    def cost_total(self, mdp_fh):
        policy_all, avg_techstages = self.aggregate_annotated_policies(mdp_fh)
        y_all = []
        for policy in policy_all:
            y_total = [mdp_fh.mdp_cost.calc_total_cost(state, a) for state, a in policy]
            y_all.append(y_total)
        y_mean, y_lower, y_upper = self.calc_data_bounds_std(y_all)
        return (y_mean, y_lower, y_upper)

    ## STATE

    def get_state_variable(self, mdp_fh, var_code):
        policy_all, avg_techstages = self.aggregate_annotated_policies(mdp_fh)
        y_all = []
        for policy in policy_all:
            y_variables = self.policy_extract_state_variable(var_code, policy)
            y_all.append(y_variables)
        y_mean, y_lower, y_upper = self.calc_data_bounds_std(y_all)
        return (y_mean, y_lower, y_upper)

    def policy_extract_state_variable(self, var_code, policy_annotated):
        idx = 0
        if var_code == 'v':
            idx = 1
        elif var_code == 'r':
            idx = 2
        elif var_code == 'l':
            idx = 3
        elif var_code == 'e':
            idx = 4
        variables = []
        for state, a in policy_annotated:
            # Include plants built in current year with total RES plants.
            if var_code == 'r':
                variables.append(state[idx]+a)
            elif var_code == 'e':
                variables.append(self.map_id_to_adjustment(state[idx]))
            else:
                variables.append(state[idx])
        return variables

    ## HELPER FUNCTIONS

    # Aggregate annotated policy for multiple runs of MDP.
    def aggregate_annotated_policies(self, mdp_fh):
        policy_all = []
        runs = self.repeat_mdp_stochastic_techstage(mdp_fh)
        for run in runs:
            # Trim annotated policy to given time range.
            policy_all.append(self.annotate_opt_policy_techstage(mdp_fh, run)[self.t0:self.tN])
        avg_techstages = np.sum(runs, axis=0)[self.t0:self.tN]/self.n_iter
        return policy_all, avg_techstages

    # Zip state with action taken in state for optimal policy of single MDP run.
    def annotate_opt_policy_techstage(self, mdp_fh, run):
        opt_policy = mdp_fh.mdp_inst.policy
        policy_annotated = []
        t = 0
        r = 0
        v = 0
        l = mdp_fh.n_tax_levels // 2
        e = 0
        for step in np.arange(0, mdp_fh.n_years):
            state_curr = (t, v, r, l, e)
            # Get tech stage from stochastic run.
            v = run[step]
            # Get updated tax level and delta emissions target.
            l, e = mdp_fh.update_state_end_of_cycle(state_curr)
            state = (t, v, r, l, e)
            idx = mdp_fh.state_to_id[state]
            a = opt_policy[idx][step]
            policy_annotated.append([(t, v, r, l, e), a])
            t += 1
            r += a
        return policy_annotated

    # Calculate mean and confidence interval of max size given a matrix where each row is a data array.
    def calc_data_bounds_max(self, data_all, axis=0):
        lower = data_all.min(1)
        upper = data_all.max(1)
        mean = np.sum(data_all, axis=axis)/self.n_iter
        return mean, lower, upper

    # Calculate mean and confidence interval of n * stdev given a matrix where each row is a data array.
    def calc_data_bounds_std(self, data_all, axis=0, n_dev=2):
        data_df = pd.DataFrame(data_all)
        std = data_df.std(axis=axis)
        mean = data_df.mean(axis=axis)
        lower = mean - n_dev*std
        upper = mean + n_dev*std
        return mean.values, lower.values, upper.values

    # Return array of years for given time range.
    def get_time_range(self, t_range):
        return np.arange(t_range[0], t_range[1])

    # Convert absolute index to tax adjustment.
    def map_id_to_adjustment(self, idx):
        if idx == 0:
            return 0
        elif idx == 1:
            return -1
        elif idx == 2:
            return 1
        else:
            raise ValueError("Invalid tax adjustment index. Expected one of {}".format([0, 1, 2]))

    # Run MDP multiple times with stochastically calculated tech stage.
    def repeat_mdp_stochastic_techstage(self, mdp_fh):
        runs = np.zeros([self.n_iter, mdp_fh.n_years], dtype=int)
        for i in np.arange(self.n_iter):
            techstage = 0
            p_adv = mdp_fh.p_adv_tech[0]
            for step in np.arange(1, mdp_fh.n_years):
                # Decide whether or not the tech stage advances for next year.
                adv = np.random.binomial(1, p_adv)
                if adv and techstage < mdp_fh.n_tech_stages - 1:
                    p_adv = mdp_fh.p_adv_tech[techstage+1]
                    techstage += 1
                runs[i][step] = techstage
        return runs


class MdpPlotter():
    def __init__(self):
        self.fig = None
        self.ax = None

    def initialize(self, x_label, y_label, title):
        self.fig, self.ax = plt.subplots()
        self.ax.set(xlabel=x_label, ylabel=y_label)
        self.ax.set_title(title)
        self.ax.grid(axis='y')

    def finalize(self):
        self.fig.tight_layout()
        return self.fig

    def plot_bars(self, x, y_bars, bar_labels, y_min=None, y_max=None,
                  colors=None, legend_loc='best', width=0.20, error=None):
        colors = self.get_colors(colors, len(bar_labels))
        for i in range(len(y_bars)):
            if error is not None:
                self.ax.bar(x+(i*width), y_bars[i], width=width, color=colors[i], label=bar_labels[i])
            else:
                self.ax.bar(x+(i*width), y_bars[i], yerr=error, width=width, color=colors[i], ecolor='k', label=bar_labels[i])
        self.set_y_range(self.ax, y_min, y_max)
        self.ax.legend(loc=legend_loc)

    def plot_heatmap(self, x, y_matrix, cmap='YlGn'):
        im = self.ax.imshow(y_matrix, cmap=cmap)
        threshold = im.norm(np.max(y_matrix))/2
        for i in range(y_matrix.shape[0]):
            for j in range(y_matrix.shape[1]):
                if int(im.norm(y_matrix[i][j])) < threshold:
                    c = 'k'
                else:
                    c = 'w'
                self.ax.text(j, i, y_matrix[i][j], ha="center", va="center", color=c)
        self.ax.set_xticks(np.arange(0, y_matrix.shape[1], 2))
        self.ax.set_yticks(np.arange(y_matrix.shape[0]))

    def plot_lines(self, x, y_lines, line_labels, y_min=None, y_max=None,
                   colors=None, legend_loc='best', y_lower=None, y_upper=None):
        colors = self.get_colors(colors, len(line_labels))
        for i in range(len(y_lines)):
            self.ax.plot(x, y_lines[i], color=colors[i], label=line_labels[i])
            if y_upper is not None and y_lower is not None:
                self.ax.fill_between(x, y_lower[i], y_upper[i], color=colors[i], alpha=0.1)
        self.set_y_range(self.ax, y_min, y_max)
        self.ax.legend(loc=legend_loc)

    def plot_stacked_bar(self, x, y_bars, bar_labels, y_min=None, y_max=None,
                         colors=None, legend_loc='best', width=0.25):
        colors = self.get_colors(colors, len(bar_labels))
        self.ax.bar(x, y_bars[0], width=width, label=bar_labels[0], color=colors[0], edgecolor='w')
        for i in np.arange(1, len(y_bars)):
            y = y_bars[i]
            bottom = np.sum(y_bars[0:i], axis=0)
            self.ax.bar(x, y, width=width, bottom=bottom, label=bar_labels[i], color=colors[i], edgecolor='w')
        self.set_y_range(self.ax, y_min, y_max)
        self.ax.legend(loc=legend_loc)

    def twin_lines(self, x, y_lines, y_label, line_labels, y_min=None, y_max=None,
                   colors=None, legend_loc='best', y_lower=None, y_upper=None, y_colors=None):
        colors = self.get_colors(colors, len(line_labels))
        self.axT = self.ax.twinx()
        for i in range(len(y_lines)):
            self.axT.plot(x, y_lines[i], color=colors[i], label=line_labels[i])
            if y_upper is not None and y_lower is not None:
                self.axT.fill_between(x, y_lower[i], y_upper[i], color=colors[i], alpha=0.1)
        self.axT.set_ylabel(y_label)
        if y_colors:
            self.ax.yaxis.label.set_color(colors[0])
            self.axT.yaxis.label.set_color(colors[1])
        self.set_y_range(self.axT, y_min, y_max)
        self.axT.legend(loc=legend_loc)

    ## HELPER FUNCTIONS

    def get_colors(self, colors, n_colors, default_colors=None, name='hsv'):
        if colors:
            return colors
        elif default_colors:
            return default_colors
        else:
            color_map = plt.cm.get_cmap(name, n_colors+1)
            return [color_map(c) for c in np.arange(n_colors)]

    def set_y_range(self, ax, y_min, y_max):
        if y_min and y_max is None:
            ax.set_ylim(bottom=y_min)
        elif y_min is None and y_max:
            ax.set_ylim(top=y_max)
        elif y_min and y_max:
            ax.set_ylim(bottom=y_min, top=y_max)
