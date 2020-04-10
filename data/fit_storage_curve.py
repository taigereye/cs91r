import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import getopt
import sys


def exp_func(x, a, b, c):
    return a * np.exp(b*x) + c


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "d:")
    except getopt.GetoptError:
        print('usage: fit_storage_curve.py -d <datafile>')
        sys.exit(2)

    with open(opts[0][1], 'r') as datafile:
        data = pd.read_csv(datafile, names=['x', 'y'], header=None)

    coef_names = ['a', 'b', 'c']
    popt, pcov = curve_fit(exp_func, data.x, data.y, bounds=(0.01, np.inf))

    print("Exponential function coefficients:")
    for c, n in zip(popt, coef_names):
        print(n, ':', c)


if __name__ == "__main__":
    main(sys.argv[1:])
