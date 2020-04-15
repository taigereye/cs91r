import getopt
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def exp_func(x, a, b, c):
    return a * np.exp(b*x) + c


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "d:")
    except getopt.GetoptError:
        print('usage: fit_storage_curve.py -d <datafile>')
        sys.exit(2)

    df = opts[0][1]
    with open(df, 'r') as datafile:
        data = pd.read_csv(datafile, names=['x', 'y'], header=None)

    coef_names = ['a', 'b', 'c']
    coefs, coef_covs = curve_fit(exp_func, data.x, data.y, bounds=(1e-6, np.inf))

    print("Exponential function coefficients:")
    for c, n in zip(coefs, coef_names):
        print(n, ':', c)

    fig, ax = plt.subplots()
    ax.scatter(data.x, data.y, color='r')
    ax.plot(np.arange(0, 100), np.apply_along_axis(exp_func, 0, np.arange(0, 100), *coefs), color='b')
    ax.set(xlabel="Renewable Penetration (%)", ylabel="Storage Capacity (% total load)")
    ax.grid()

    fig.savefig(df.split('.')[0] + ".png")
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
