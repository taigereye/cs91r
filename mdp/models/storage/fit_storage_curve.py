import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.optimize import curve_fit


def exp_func(x, a, b):
    return a * np.exp(b*x) - a


def main(argv):
    parser = argparse.ArgumentParser(description="fit exponential function calculating storage based on renewable penetration")
    parser.add_argument("-d", "--datafile", help="csv file with renewable penetration and resulting storage fractions")
    args = parser.parse_args()

    df = args.datafile
    with open(df, 'r') as datafile:
        data = pd.read_csv(datafile, names=['x', 'y'], header=None)

    data.x *= 100
    data.y *= 100

    coef_names = ['a', 'b']
    coefs, coef_covs = curve_fit(exp_func, data.x, data.y, bounds=(1e-6, np.inf))

    of = Path(df.split('.')[0] + ".txt")
    with open(of, 'w+') as outfile:
        outfile.write("Exponential function coefficients:\n")
        for c, n in zip(coefs, coef_names):
            outfile.write("{}:{}\n".format(n, c))

    fig, ax = plt.subplots()
    ax.scatter(data.x, data.y, color='r')
    ax.plot(np.arange(0, 100), np.apply_along_axis(exp_func, 0, np.arange(0, 100), *coefs), color='b')
    ax.set(xlabel="Renewable Penetration (%)", ylabel="Storage Capacity (% total load)")
    ax.grid()
    fig.savefig(df.split('.')[0] + ".png")
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
