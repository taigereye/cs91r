import argparse
import sys

from pathlib import Path


def main(argv):
    parser = argparse.ArgumentParser(description="generate one script per job to collect MDP data")
    parser.add_argument("-p", "--paramslist", help="txt file with newline separated list of paramsfiles")
    parser.add_argument("-c", "--cores", help="allotted cores on single machine", type=int, default=8)
    parser.add_argument("-t", "--hrs", help="total allotted runtime in hours", type=int, default=8)
    parser.add_argument("-g", "--gigs", help="allotted memory in GB", type=int, default=128)
    args = parser.parse_args()

    fasrc_dir = Path("fasrc/")
    pf = fasrc_dir / "{}.txt".format(args.paramslist)
    with open(pf, 'r') as paramsfile:
        params_list = paramsfile.readlines()
    paramsfile.close()
    params_list = [p.strip() for p in params_list]

    for paramsfile in params_list:
        script_dir = Path("fasrc/scripts")
        sf = script_dir / "run_{}.sh".format(paramsfile)
        with open(sf, 'w+') as scriptfile:
            scriptfile.write("#!/bin/bash\n")
            scriptfile.write("#SBATCH -n {:d}    # Number of cores (-n)\n")
            scriptfile.write("#SBATCH -N 2    # Ensure that all cores are on one Node (-N)\n")
            scriptfile.write("#SBATCH -t 0-{:02d}:00    # Runtime in D-HH:MM, minimum of 10 minutes\n")
            scriptfile.write("#SBATCH -p tambe    # Partition to submit to")
            scriptfile.write("#SBATCH --mem={:d}000    # Memory pool for all cores (see also --mem-per-cpu)\n")
            scriptfile.write("#SBATCH -o myoutput_%j.out    # File to which STDOUT will be written, %j inserts jobid\n")
            scriptfile.write("#SBATCH -e myerrors_%j.err     # File to which STDERR will be written, %j inserts j\n")
            scriptfile.write("module load Anaconda3/2019.10\n")
            scriptfile.write("python fill_mdp_fh_data.py -p {}\n".format(args.cores,
                                                                         args.hrs,
                                                                         args.gigs,
                                                                         paramsfile))
        scriptfile.close()


if __name__ == "__main__":
    main(sys.argv[1:])