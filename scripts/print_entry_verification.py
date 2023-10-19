import argparse
import json

import numpy as np
from tqdm import tqdm

from value_disagreement.extraction import ValueConstants
from value_disagreement.vizualization.profile_radar import plot_radar


def main(args):
    with open(args.infile_profiles, "r") as f:
        data_profiles = json.load(f)
    users = list(data_profiles['profiles'].keys())

    if args.n_prints < 0:
        usernames = users
    else:
        usernames = np.random.choice(users, size=args.n_prints, replace=False)

    for username in tqdm(usernames, total=len(usernames)):
        values = ValueConstants.SCHWARTZ_VALUES
        profile = np.array(data_profiles['profiles'][username])
        profile_max = np.max(profile)
        normed_profile = profile / profile_max
        data = (f"Schwartz values user={username}", [normed_profile.tolist()])
        plot_radar(values, data, outname=f"output/profile_images/profile_user={username}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Plot a sample of the the given profiles on a radar plot")
    parser.add_argument('infile_profiles', help="Which file to load the profiles from")
    parser.add_argument('--n_prints', type=int, default=10, help="how many users to print")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")

    main(args)