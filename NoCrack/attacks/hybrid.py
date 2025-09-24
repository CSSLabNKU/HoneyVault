import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
nle_dir = os.path.dirname(curr_dir)
sys.path.append(nle_dir)

import random
import argparse
from tqdm import tqdm
from collections import Counter
from toolkits import load_pickle, save_json
from password_similarity import (
    feature_functions,
    feature_difference_distribution,
    password_similarity,
)
from single_password import single_password
from nocrack_nle import init_nocrack_nle


def sign(value):
    """
    Returns 1 if the value is positive, -1 if the value is negative, and 0 if the value is zero.
    """
    if value > 0:
        return 1
    elif value < 0:
        return -1
    else:
        return 0


def hybrid(f_a, Pr_decoy, D_size, diff_distri, real_distribution, decoy_distribution):
    """USENIX'21 Cheng et al.
    single_password and password_similarity
    """
    sp_score = single_password(f_a, Pr_decoy, D_size)
    ps_score = password_similarity(diff_distri, real_distribution, decoy_distribution)
    return sp_score * ps_score


def parse_command():
    parser = argparse.ArgumentParser(description="KL divergence attack")
    parser.add_argument(
        "--num",
        required=False,
        type=int,
        default=100,
        help="Number of sweetvaults.",
    )
    parser.add_argument(
        "--M",
        required=False,
        type=str,
        default="CM",
        help="Feature for decoy vaults",
    )
    parser.add_argument(
        "--I",
        required=False,
        type=str,
        default="LCSStr",
        help="Feature for real vaults",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default="wishbone",
        help="Dataset: wishbone, rockyou",
    )
    return parser.parse_args()


def main(args):
    nle = init_nocrack_nle()
    sweet_vaults = load_pickle(f"{nle_dir}/results/sweetvaults_{args.num}.pkl")
    D = load_pickle(f"{nle_dir}/data/{args.dataset}_counter.pkl")
    D_size = sum(D.values())

    feature_dir = os.path.join(nle_dir, "results")
    feature_difference_file = os.path.join(feature_dir, f"{args.M}_and_{args.I}.pkl")
    if not os.path.exists(feature_difference_file):
        feature_difference_distribution(
            feature_functions[args.M],
            feature_functions[args.I],
            sweet_vaults,
            nle,
        )

    feature_difference_distri = load_pickle(feature_difference_file)

    real_distribution = {
        "M_I": {1: 0, 0: 0, "_total_": 0},
        "I_M": {1: 0, 0: 0, "_total_": 0},
    }
    decoy_distribution = {
        "M_I": {1: 0, 0: 0, "_total_": 0},
        "I_M": {1: 0, 0: 0, "_total_": 0},
    }

    for id, diff_distri in feature_difference_distri.items():
        for i, diff in enumerate(diff_distri):
            if i == 0:
                real_distribution["M_I"][diff[0]] += 1
                real_distribution["I_M"][diff[1]] += 1
                real_distribution["M_I"]["_total_"] += 1
                real_distribution["I_M"]["_total_"] += 1
            else:
                decoy_distribution["M_I"][diff[0]] += 1
                decoy_distribution["I_M"][diff[1]] += 1
                decoy_distribution["M_I"]["_total_"] += 1
                decoy_distribution["I_M"]["_total_"] += 1

    ranks = {}
    for id in tqdm(sweet_vaults.keys(), total=len(sweet_vaults.keys())):
        diff_distri = feature_difference_distri[id]
        vaults = sweet_vaults[id]
        score_list = []

        for i in range(args.num):
            vault = vaults[i]
            vault_counter = Counter(vault)
            Pr_decoy = []
            f_a = []

            for pw, freq in vault_counter.items():
                pr_real = D.get(pw, 0)
                pr_decoy = nle.spm.prob(pw, "IS-DTE")
                f_a.extend([pr_real] * freq)
                Pr_decoy.extend([pr_decoy] * freq)

            score = hybrid(
                f_a,
                Pr_decoy,
                D_size,
                diff_distri[i],
                real_distribution,
                decoy_distribution,
            )

            if i == 0:
                score_list.append((True, score))
            else:
                score_list.append((False, score))

        random.shuffle(score_list)
        score_list = sorted(score_list, key=lambda x: x[1], reverse=True)
        index = next((k for k, t in enumerate(score_list) if t[0] == True), None)
        ranks[id] = index

    results_dir = os.path.join(nle_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    save_json(ranks, f"{results_dir}/hybrid_ranks_{args.num}.json", 4)


if __name__ == "__main__":
    args = parse_command()
    main(args)

# python hybrid.py --num 100 --dataset wishbone --M CM --I LCSStr
