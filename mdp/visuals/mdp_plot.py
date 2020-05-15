import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


def plot_heatmap(x, y_2D, y_max, x_label, y_label, title, cmap=None):
    fig, ax = plt.subplots()
    if not cmap:
        cmap = "YlGn"
    im = ax.imshow(y_2D, cmap=cmap)
    threshold = im.norm(np.max(y_2D))/2
    for i in range(y_2D.shape[0]):
        for j in range(y_2D.shape[1]):
            if int(im.norm(y_2D[i][j])) < threshold:
                c = 'k'
            else:
                c = 'w'
            text = ax.text(j, i, y_2D[i][j], ha="center", va="center", color=c)
    ax.set(xlabel=x_label, ylabel=y_label)
    ax.set_xticks(np.arange(0, y_2D.shape[1], 2))
    ax.set_yticks(np.arange(y_2D.shape[0]))
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_multiple_bar(x, y_all, x_label, y_label, labels, title, w=0.20, colors=None):
    if not colors:
        color_map = get_color_map(y_all.shape[0])
        colors = [color_map(c) for c in np.arange(y_all.shape[0])]
    fig, ax = plt.subplots()
    for i in np.arange(len(y_all)):
        y = y_all[i]
        ax.bar(x+(i*w), y, width=w-0.05, color=colors[i], edgecolor='w', label=labels[i])
    ax.grid(axis='y')
    ax.set(xlabel=x_label, ylabel=y_label)
    ax.legend(loc='best', labels=labels)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_multiple_bar_double(x, y_pair_all, x_label, y_label, title, w=0.20, colors=None):
    if not colors:
        color_map = get_color_map(y_pair_all.shape[0])
        colors = [color_map(c) for c in np.arange(y_pair_all.shape[0])]
    fig, ax = plt.subplots()
    for i in np.arange(y_pair_all.shape[0]):
        y_pair = y_pair_all[i]
        ax.bar(x+(i*3/2*w), y_pair[0], width=w/2, color=colors[i], edgecolor='w')
        ax.bar(x+(i*3/2*w)+w/2, y_pair[1], width=w/2, color=colors[i], edgecolor='w', alpha=0.50)
    ax.grid(axis='y')
    ax.set(xlabel=x_label, ylabel=y_label)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_multiple_bar_stacked(x, y_all_v, x_label, y_label, labels, title, w=0.15, colors=None, percent=False):
    if not colors:
        color_map = get_color_map(len(labels))
        colors = [color_map(c) for c in np.arange(y_all_v.shape[1])]
    fig, ax = plt.subplots()
    for i in np.arange(y_all_v.shape[0]):
        y_all = y_all_v[i]
        if percent:
            y_all = y_all / np.sum(y_all, axis=0)
        label0 = (labels[0] if i == 0 else "")
        ax.bar(x+(i*w), y_all[0], width=w-0.05, label=label0, color=colors[0], edgecolor='w')
        for j in np.arange(1, y_all_v.shape[1]):
            y = y_all[j]
            labelj = (labels[j] if i == 0 else "")
            ax.bar(x+(i*w), y, width=w-0.05, bottom=np.sum(y_all[0:j], axis=0), label=labelj, color=colors[j], edgecolor='w')
    ax.grid(axis='y')
    ax.set(xlabel=x_label, ylabel=y_label)
    ax.legend(loc='best', )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_multiple_bar_stacked_double(x, y_pair_all_v, x_label, y_label, labels, title, w=0.20, colors=None, percent=False):
    if not colors:
        color_map = get_color_map(len(labels))
        colors = [color_map(c) for c in np.arange(y_pair_all_v.shape[2])]
    fig, ax = plt.subplots()
    for i in np.arange(y_pair_all_v.shape[0]):
        y0_all = y_pair_all_v[i][0]
        y1_all = y_pair_all_v[i][1]
        if percent:
            y0_all = y0_all / np.sum(y0_all, axis=0)
            y1_all = y1_all / np.sum(y1_all, axis=0)
        label0 = (labels[0] if i == 0 else "")
        ax.bar(x+(i*3/2*w), y0_all[0], width=w/2, label=label0, color=colors[0], edgecolor='w')
        ax.bar(x+(i*3/2*w)+w/2, y1_all[0], width=w/2, label=label0, color=colors[0], edgecolor='w', alpha=0.50)
        for j in np.arange(1, y_pair_all_v.shape[2]):
            y0 = y0_all[j]
            y1 = y1_all[j]
            labelj = (labels[j] if i == 0 else "")
            ax.bar(x+(i*3/2*w), y0, width=w/2, bottom=np.sum(y0_all[0:j], axis=0), label=labelj, color=colors[j], edgecolor='w')
            ax.bar(x+(i*3/2*w)+w/2, y1, width=w/2, bottom=np.sum(y1_all[0:j], axis=0), label=labelj, color=colors[j], edgecolor='w', alpha=0.50)
    ax.grid(axis='y')
    ax.set(xlabel=x_label, ylabel=y_label)
    ax.legend(loc='best')
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_multiple_bar_twin_multiple_line(x, y_lines, y_bars, x_label, y_label_line, y_label_bar, labels_lines, labels_bars, title, w=0.30, colors=None):
    if not colors:
        color_map = get_color_map(len(y_lines))
        colors = [color_map(c) for c in np.arange(len(y_lines))]
    fig, ax = plt.subplots()
    for i in np.arange(len(y_bars)):
        ax.bar(x+i*w, y_bars[i], width=0.75*w, color=colors[i], label=labels_bars[i], alpha=0.50)
    ax.grid(axis='y')
    ax.set(xlabel=x_label, ylabel=y_label_bar)
    axT = ax.twinx()
    for i in np.arange(len(y_lines)):
        axT.plot(x, y_lines[i], color=colors[i], label=labels_lines[i])
    axT.set_ylabel(y_label_line)
    axT.legend(loc='lower right')
    ax.legend(loc='upper left')
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_multiple_line(x, y_all, y_max, x_label, y_label, labels, title, scalar, scalar_name=None, colors=None, is_fixed=False):
    if not colors:
        color_map = get_color_map(len(labels))
        colors = [color_map(c) for c in np.arange(y_all.shape[0])]
    fig, ax = plt.subplots()
    for i in np.arange(y_all.shape[0]):
        ax.plot(x, y_all[i], color=colors[i], label=labels[i])
    if scalar:
        if is_fixed:
            ax.axhline(y=scalar, color='k', linestyle='dashed', label=scalar_name)
        else:
            y_scalar = np.cumsum(np.full(y_all.shape[1], scalar))
            ax.plot(x, y_scalar, color='k', linestyle='dashed', label=scalar_name)
    ax.grid(axis='y')
    ax.set(xlabel=x_label, ylabel=y_label)
    if y_max:
        ax.set_ylim(0, y_max)
    ax.set_ylim(bottom=0)
    ax.legend(loc='lower right')
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_multiple_line_twin_single_bar(x, y_lines, y_bar, y_bar_max, x_label, y_label_lines, y_label_bar, labels, title, colors=None, is_annual=False):
    if not colors:
        colors = ['g', 'r', 'b', 'darkred', 'midnightblue']
    if is_annual:
        c = 1
    else:
        c = 3
    fig, ax = plt.subplots()
    ll = ax.plot(x, y_lines[0], color=colors[c+0], label=labels[1])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label_lines[0])
    ax.yaxis.label.set_color(colors[c+0])
    axT = ax.twinx()
    lr = axT.plot(x, y_lines[1], color=colors[c+1], label=labels[2])
    axT.set_ylabel(y_label_lines[1])
    axT.yaxis.label.set_color(colors[c+1])
    axB = ax.twinx()
    lb = axB.bar(x, y_bar, width=0.10, color=colors[0], label=labels[0], alpha=0.75)
    axB.spines["right"].set_position(("axes", 1.1))
    axB.spines["right"].set_visible(True)
    axB.set_ylabel(y_label_bar)
    axB.set_ylim(0, y_bar_max)
    axB.yaxis.label.set_color(colors[0])
    axB.legend(loc='best')
    lines = ll+lr
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_single_bar(x, y, x_label, y_label, title, w=0.30, color=None, bar_labels=None):
    if not color:
        color = 'gray'
    fig, ax = plt.subplots()
    ax.bar(x, y, width=w, color=color, edgecolor='w')
    if bar_labels is not None:
        label_single_bars_above(ax.patches, bar_labels, ax)
    ax.grid(axis='y')
    ax.set(xlabel=x_label, ylabel=y_label)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_single_bar_double(x, y_pair, x_label, y_label, title, w=0.30, color=None, bar_labels=None):
    if not color:
        color = 'k'
    fig, ax = plt.subplots()
    ax.bar(x, y_pair[0], width=w/2, color=color, edgecolor='w')
    ax.bar(x+w/2, y_pair[1], width=w/2, color=color, edgecolor='w', alpha=0.50)
    if bar_labels is not None:
        label_single_bars_above(ax.patches, bar_labels, ax)
    ax.grid(axis='y')
    ax.set(xlabel=x_label, ylabel=y_label)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_single_bar_double_twin_line(x, y_bar, y_line, x_label, y_label_bar, y_label_line, labels, title, w=0.30, colors=None):
    if not colors:
        colors = ['b', 'g']
    fig, ax = plt.subplots()
    ax.bar(x, y_bar[0], width=w, color=colors[0], edgecolor='w', label=labels[0])
    ax.bar(x+w/2, y_bar[1], width=w/2, color=colors[0], edgecolor='w', label=labels[1], alpha=0.50)
    ax.grid(axis='y')
    ax.set(xlabel=x_label, ylabel=y_label_bar)
    ax.yaxis.label.set_color(colors[1])
    axT = ax.twinx()
    axT.plot(x, y_line, color=colors[0])
    axT.set_ylabel(y_label_line)
    axT.yaxis.label.set_color(colors[0])
    ax.legend(loc='best')
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_single_bar_stacked(x, y_all, x_label, y_label, labels, title, w=0.30, colors=None, percent=False):
    if not colors:
        color_map = get_color_map(len(labels))
        colors = [color_map(c) for c in np.arange(len(y_all))]
    fig, ax = plt.subplots()
    if percent:
        y_all = y_all / np.sum(y_all, axis=0)
    ax.bar(x, y_all[0], width=w, label=labels[0], color=colors[0], edgecolor='w')
    for i in np.arange(1, len(y_all)):
        y = y_all[i]
        ax.bar(x, y, width=w, bottom=np.sum(y_all[0:i], axis=0), label=labels[i], color=colors[i], edgecolor='w')
    ax.grid(axis='y')
    ax.set(xlabel=x_label, ylabel=y_label)
    ax.set_ylim(top=1.0)
    ax.legend(loc='best')
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_single_bar_stacked_double(x, y_pair_all, x_label, y_label, labels, title, w=0.30, colors=None, percent=False):
    if not colors:
        color_map = get_color_map(len(labels))
        colors = [color_map(c) for c in np.arange(len(y_pair_all[0]))]
    fig, ax = plt.subplots()
    y0_all = y_pair_all[0]
    y1_all = y_pair_all[1]
    if percent:
        y0_all = y0_all / np.sum(y0_all, axis=0)
        y1_all = y1_all / np.sum(y1_all, axis=0)
    ax.bar(x, y0_all[0], width=w/2, label=labels[0], color=colors[0], edgecolor='w')
    ax.bar(x+w/2, y1_all[0], width=w/2, label=labels[0], color=colors[0], edgecolor='w', alpha=0.50)
    for i in np.arange(1, len(y_pair_all[0])):
        ax.bar(x, y0_all[i], width=w/2, bottom=np.sum(y0_all[0:i], axis=0), label=labels[i], color=colors[i], edgecolor='w')
        ax.bar(x+w/2, y1_all[i], width=w/2, bottom=np.sum(y1_all[0:i], axis=0), label=labels[i], color=colors[i], edgecolor='w', alpha=0.50)
    ax.grid(axis='y')
    ax.set(xlabel=x_label, ylabel=y_label)
    ax.legend(loc='best')
    ax.set_title(title)
    fig.tight_layout()
    return fig


# HELPER FUNCTIONS


def get_color_map(n, name='hsv'):
    return plt.cm.get_cmap(name, n+1)


def get_legend_handles(labels):
    legend_handles = []
    for l in labels:
        handle = Patch(label=l)
        legend_handles.append(handle)
    return legend_handles


def label_single_bars_above(rects, bar_labels, ax):
    for rect, label in zip(rects, bar_labels):
        x = rect.get_x() + rect.get_width()/2.
        y = rect.get_height()
        ax.annotate("{}".format(label), (x, y), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom')


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
