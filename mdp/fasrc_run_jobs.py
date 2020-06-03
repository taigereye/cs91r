import argparse
import subprocess
import sys

from pathlib import Path


def main(argv):
    parser = argparse.ArgumentParser(description="collect MDP data for list of paramsfiles")
    parser.add_argument("-p", "--paramslist", help="txt file with newline separated list of paramsfiles")
    args = parser.parse_args()

    fasrc_dir = Path("fasrc/")
    pf = fasrc_dir / "{}.txt".format(args.paramslist)
    with open(pf, 'r') as paramsfile:
        params_list = paramsfile.readlines()
    paramsfile.close()
    params_list = [p.strip() for p in params_list]

    for paramsfile in params_list:
        subprocess.call(['python', 'fill_mdp_fh_data.py', '-p', paramsfile])


if __name__ == "__main__":
    main(sys.argv[1:])