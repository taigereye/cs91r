from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MdpDataGatherer():
    def __init__(self, mdp_model, n_iter, t_range, ci_type="QRT", n_dev=1, q_lower=0.25, q_upper=0.75):
        self.mdp_model = mdp_model
        self.instances = OrderedDict()
        self.n_iter = n_iter
        self.t0 = t_range[0]
        self.tN = t_range[1]
        self.start_year = 2020
        self.ci_type = ci_type
        self.n_dev = n_dev
        self.q_lower = q_lower
        self.q_upper = q_upper

    def add_mdp_instance(self, paramsfile, params):
        assert(self.mdp_model.param_names == list(params.keys()))
        mdp_fh = self.mdp_model.run_fh(params)
        self.instances[paramsfile] = mdp_fh
        return mdp_fh

    def set_ci_type(self, ci_type, n_dev=1, q_lower=0.25, q_upper=0.75):
        self.ci_type = ci_type
        self.n_dev = n_dev
        self.q_lower = q_lower
        self.q_upper = q_upper

    ## COST

    # Get all cost components averaged across stochastic tech stage.
    def cost_breakdown_components(self, mdp_fh, components, is_percent=False):
        policy_all, avg_techstages = self._aggregate_annotated_policies(mdp_fh)
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
        policy_all, avg_techstages = self._aggregate_annotated_policies(mdp_fh)
        y_all = []
        for policy in policy_all:
            y_component = [mdp_fh.mdp_cost.calc_partial_cost(state, a, component) for state, a in policy]
            y_all.append(y_component)
        return self.calc_data_bounds(y_all)

    # Get total cost averaged across stochastic tech stage.
    def cost_total(self, mdp_fh):
        policy_all, avg_techstages = self._aggregate_annotated_policies(mdp_fh)
        y_all = []
        for policy in policy_all:
            y_total = [mdp_fh.mdp_cost.calc_total_cost(state, a) for state, a in policy]
            y_all.append(y_total)
        return self.calc_data_bounds(y_all)

    ## CO2

    def co2_current_price(self, mdp_fh):
        policy_all, avg_techstages = self._aggregate_annotated_policies(mdp_fh)
        y_all = []
        for policy in policy_all:
            y_ff = self._get_ff_plants(policy, mdp_fh.n_plants)
            y_price = [mdp_fh.mdp_cost.co2_price(t, l, f) for ((t, v, r, l, e), a), f in zip(policy, y_ff)]
            y_all.append(y_price)
        return self.calc_data_bounds(y_all)

    def co2_emissions(self, mdp_fh):
        policy_all, avg_techstages = self._aggregate_annotated_policies(mdp_fh)
        y_all = []
        for policy in policy_all:
            y_ff = self._get_ff_plants(policy, mdp_fh.n_plants)
            y_price = [mdp_fh.mdp_cost.co2_emit(f) for f in y_ff]
            y_all.append(y_price)
        return self.calc_data_bounds(y_all)

    def co2_tax_collected(self, mdp_fh):
        policy_all, avg_techstages = self._aggregate_annotated_policies(mdp_fh)
        y_all = []
        for policy in policy_all:
            y_ff = self._get_ff_plants(policy, mdp_fh.n_plants)
            y_price = [mdp_fh.mdp_cost.co2_tax(t, l, f) for ((t, v, r, l, e), a), f in zip(policy, y_ff)]
            y_all.append(y_price)
        return self.calc_data_bounds(y_all)

    def target_emissions(self, mdp_fh):
        emit_steps = []
        for i in range(mdp_fh.n_years):
            idx = i // mdp_fh.co2_tax_cycle
            emit_steps.append(mdp_fh.emit_targets[idx])
        return emit_steps

    ## RES

    def res_penetration(self, mdp_fh):
        policy_all, avg_techstages = self._aggregate_annotated_policies(mdp_fh)
        y_all = []
        for policy in policy_all:
            tax_levels = self.get_tax_levels(mdp_fh)
            y_variables = self._policy_extract_state_variable('r', policy, tax_levels)
            y_variables = np.asarray(y_variables) / mdp_fh.n_plants
            y_all.append(y_variables)
        return self.calc_data_bounds(y_all)

    ## STATE

    def get_state_variable(self, mdp_fh, var_code):
        policy_all, avg_techstages = self._aggregate_annotated_policies(mdp_fh)
        y_all = []
        for policy in policy_all:
            tax_levels = self.get_tax_levels(mdp_fh)
            y_variables = self._policy_extract_state_variable(var_code, policy, tax_levels)
            y_all.append(y_variables)
        y_mean = np.sum(y_all, axis=0)/self.n_iter
        return y_mean

    ## HELPER FUNCTIONS

    # Aggregate annotated policy for multiple runs of MDP.
    def _aggregate_annotated_policies(self, mdp_fh):
        policy_all = []
        runs = self._repeat_mdp_stochastic_techstage(mdp_fh)
        for run in runs:
            # Trim annotated policy to given time range.
            policy_all.append(self._annotate_opt_policy_techstage(mdp_fh, run)[self.t0:self.tN])
        avg_techstages = np.sum(runs, axis=0)[self.t0:self.tN]/self.n_iter
        return policy_all, avg_techstages

    # Zip state with action taken in state for optimal policy of single MDP run.
    def _annotate_opt_policy_techstage(self, mdp_fh, run):
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
    def calc_data_bounds(self, data_all, axis=0):
        data = dict()
        if self.ci_type == "ABS":
            mean, lower, upper = self._calc_data_bounds_abs(data_all, axis)
        elif self.ci_type == "QRT":
            mean, lower, upper = self._calc_data_bounds_qrt(data_all, axis, self.q_lower, self.q_upper)
        elif self.ci_type == "STD":
            mean, lower, upper = self._calc_data_bounds_std(data_all, axis, self.n_dev)
        else:
            raise ValueError("ci_type must be ABS or STD: {}".format(self.ci_type))
        data['mean'] = mean
        data['lower'] = lower
        data['upper'] = upper
        return data

    # Maximum upper and lower bounds.
    def _calc_data_bounds_abs(self, data_all, axis):
        lower = np.asarray(data_all).min(0)
        upper = np.asarray(data_all).max(0)
        mean = np.sum(data_all, axis=axis)/self.n_iter
        return mean, lower, upper

    # Quartile bounds.
    def _calc_data_bounds_qrt(self, data_all, axis, q_lower, q_upper):
        data_df = pd.DataFrame(data_all)
        mean = data_df.mean(axis=axis)
        lower = data_df.quantile(q=q_lower, axis=axis)
        upper = data_df.quantile(q=q_upper, axis=axis)
        return mean.values, lower.values, upper.values

    # Standard deviation bounds.
    def _calc_data_bounds_std(self, data_all, axis, n_dev):
        data_df = pd.DataFrame(data_all)
        std = data_df.std(axis=axis)
        mean = data_df.mean(axis=axis)
        lower = mean - n_dev*std
        upper = mean + n_dev*std
        return mean.values, lower.values, upper.values

    # Get FF plants based on state variables as array.
    def _get_ff_plants(self, policy_annotated, n_plants):
        ff_plants = []
        for state, a in policy_annotated:
            r = state[2]
            f = n_plants - (r+a)
            ff_plants.append(f)
        return ff_plants

    # Get tax levels for either base price or tax rate.
    def get_tax_levels(self, mdp_fh):
        if mdp_fh.mdp_cost.co2_tax_adjust == "BASE":
            return mdp_fh.mdp_cost.c_co2_base_levels
        elif mdp_fh.mdp_cost.co2_tax_adjust == "INC":
            return mdp_fh.mdp_cost.c_co2_inc_levels
        else:
            raise ValueError("co2_tax_adjust must be BASE or INC: {}".format(self.co2_tax_type))

    # Return array of years for given time range.
    def get_time_range(self, t_range):
        return np.arange(t_range[0]+self.start_year, t_range[1]+self.start_year)

    # Convert absolute index to tax adjustment.
    def _map_id_to_adjustment(self, idx):
        if idx == 0:
            return 0
        elif idx == 1:
            return -1
        elif idx == 2:
            return 1
        else:
            raise ValueError("Invalid tax adjustment index. Expected one of {}".format([0, 1, 2]))

    # Convert absolute index to tax level.
    def _map_id_to_tax_level(self, idx, tax_levels):
        tax_levels_centered = np.array(tax_levels) - tax_levels[len(tax_levels)//2]
        # tax_levels_centered = np.array(tax_levels)
        return tax_levels_centered[idx]

    # Get single state variable as array.
    def _policy_extract_state_variable(self, var_code, policy_annotated, tax_levels):
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
            elif var_code == 'l':
                variables.append(self._map_id_to_tax_level(state[idx], tax_levels))
            elif var_code == 'e':
                variables.append(self._map_id_to_adjustment(state[idx]))
            else:
                variables.append(state[idx])
        return variables

    # Run MDP multiple times with stochastically calculated tech stage.
    def _repeat_mdp_stochastic_techstage(self, mdp_fh):
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

    def initialize(self, title, x_label, y_label,
                   x_ticks=None, y_ticks=None, x_tick_labels=None, y_tick_labels=None):
        self.fig, self.ax = plt.subplots()
        self.ax.set_title(title)
        self.ax.set(xlabel=x_label, ylabel=y_label)
        self.ax.grid(axis='y')
        if x_ticks is not None:
            self.ax.set_xticks(x_ticks)
        if y_ticks is not None:
            self.ax.set_yticks(y_ticks)
        if x_tick_labels is not None:
            self.ax.set_xticklabels(x_tick_labels)
        if y_tick_labels is not None:
            self.ax.set_yticklabels(y_tick_labels)

    def finalize(self):
        self.fig.tight_layout()
        return self.fig

    ## PLOTS

    def plot_bars(self, x, y_bars, bar_labels, y_min=None, y_max=None,
                  colors=None, legend_loc='best', width=0.20, error=None):
        colors = self._get_colors(colors, len(bar_labels))
        for i in range(len(y_bars)):
            if error is not None:
                self.ax.bar(x+(i*width), y_bars[i]['mean'], width=width, color=colors[i], label=bar_labels[i])
            else:
                self.ax.bar(x+(i*width), y_bars[i], yerr=error, width=width, color=colors[i], ecolor='k', label=bar_labels[i])
        self._set_y_range(self.ax, y_min, y_max)
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
                   colors=None, legend_loc='best', CI=False):
        colors = self._get_colors(colors, len(line_labels))
        for i in range(len(y_lines)):
            self.ax.plot(x, y_lines[i]['mean'], color=colors[i], label=line_labels[i])
            if CI:
                self.ax.fill_between(x, y_lines[i]['lower'], y_lines[i]['upper'], color=colors[i], alpha=0.1)
        self._set_y_range(self.ax, y_min, y_max)
        self.ax.legend(loc=legend_loc)

    def plot_scatter(self, x, y_scatter, scatter_labels, y_min=None, y_max=None,
                     colors=None, legend_loc='best', marker='.'):
        colors = self._get_colors(colors, len(scatter_labels))
        for i in range(len(y_scatter)):
            self.ax.scatter(x, y_scatter[i], color=colors[i], marker=marker, label=scatter_labels[i])
        self._set_y_range(self.ax, y_min, y_max)
        self.ax.legend(loc=legend_loc)

    def plot_stacked_bar(self, x, y_bars, bar_labels, y_min=None, y_max=None,
                         colors=None, legend_loc='best', width=0.25):
        colors = self._get_colors(colors, len(bar_labels))
        self.ax.bar(x, y_bars[0], width=width, label=bar_labels[0], color=colors[0], edgecolor='w')
        for i in np.arange(1, len(y_bars)):
            y = y_bars[i]
            bottom = np.sum(y_bars[0:i], axis=0)
            self.ax.bar(x, y, width=width, bottom=bottom, label=bar_labels[i], color=colors[i], edgecolor='w')
        self._set_y_range(self.ax, y_min, y_max)
        self.ax.legend(loc=legend_loc)

    ## ADD ONS

    def add_fixed_line(self, x, y, label, color=None, is_dashed=False):
        if not color:
            color = 'k'
        if is_dashed:
            self.ax.plot(x, y, color=color, label=label, linestyle='dashed')
        else:
            self.ax.plot(x, y, color=color, label=label)

    def add_scatter_points(self, x, y, label, color=None, marker=None):
        if not color:
            color = 'k'
        if marker:
            self.ax.scatter(x, y, color=color, label=label, marker='dashed')
        else:
            self.ax.scatter(x, y, color=color, label=label)

    def add_twin_lines(self, x, y_lines, y_label, line_labels, y_min=None, y_max=None,
                       colors=None, legend_loc='best', CI=False, y_colors=None):
        colors = self._get_colors(colors, len(line_labels))
        self.axT = self.ax.twinx()
        for i in range(len(y_lines)):
            self.axT.plot(x, y_lines[i]['mean'], color=colors[i], label=line_labels[i])
            if CI:
                self.axT.fill_between(x, y_lines[i]['lower'], y_lines[i]['upper'], color=colors[i], alpha=0.1)
        self.axT.set_ylabel(y_label)
        if y_colors:
            self.ax.yaxis.label.set_color(colors[0])
            self.axT.yaxis.label.set_color(colors[1])
        self._set_y_range(self.axT, y_min, y_max)
        self.axT.legend(loc=legend_loc)

    ## HELPER FUNCTIONS

    def _get_colors(self, colors, n_colors, default_colors=None, name='hsv'):
        if colors:
            return colors
        elif default_colors:
            return default_colors
        else:
            color_map = plt.cm.get_cmap(name, n_colors+1)
            return [color_map(c) for c in np.arange(n_colors)]

    def _set_y_range(self, ax, y_min, y_max):
        if y_min is not None and y_max is not None:
            ax.set_ylim(bottom=y_min, top=y_max)
            return
        elif y_min is not None and y_max is None:
            ax.set_ylim(bottom=y_min)
            return
        elif y_min is None and y_max is not None:
            ax.set_ylim(top=y_max)
            return
        else:
            return
