"""
Adaptive hybrid attack against VaultGuard NLE.
"""

import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
nle_dir = os.path.dirname(curr_dir)
sys.path.append(nle_dir)

import random
import argparse
from tqdm import tqdm
from collections import Counter
from vaultguard_nle import init_vaultguard_nle
from toolkits import load_pickle, save_json
from password_similarity import (
    feature_functions,
    feature_difference_distribution,
    password_similarity,
)
from single_password import single_password
from adaptive_extra import adaptive_extra


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
    sp_score = single_password(f_a, Pr_decoy, D_size)
    ps_score = password_similarity(diff_distri, real_distribution, decoy_distribution)
    return sp_score * ps_score


def adaptive_hybrid(
    f_a,
    Pr_decoy,
    D_size,
    diff_distri,
    real_distribution,
    decoy_distribution,
    vault,
    boost,
):
    hybrid_score = hybrid(
        f_a, Pr_decoy, D_size, diff_distri, real_distribution, decoy_distribution
    )
    ae_score = adaptive_extra(vault, boost)
    return sign(ae_score) * hybrid_score


def parse_command():
    parser = argparse.ArgumentParser(description="Adaptive hybrid attack")
    parser.add_argument(
        "--num",
        required=False,
        type=int,
        default=1000,
        help="Number of sweetvaults.",
    )
    parser.add_argument(
        "--pseudocount",
        type=float,
        required=False,
        default=1.000,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        required=False,
    )
    parser.add_argument(
        "--M",
        required=False,
        type=str,
        default="LCSStr",
        help="Feature for decoy vaults",
    )
    parser.add_argument(
        "--I",
        required=False,
        type=str,
        default="LCS",
        help="Feature for real vaults",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default="wishbone",
    )
    return parser.parse_args()


def main(args):
    nle = init_vaultguard_nle()
    sweet_vaults = load_pickle(f"{nle_dir}/results/sweetvaults_{args.num}.pkl")
    sweet_reuses = load_pickle(f"{nle_dir}/results/sweetreuses_{args.num}.pkl")
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
    for id in tqdm(
        sweet_vaults.keys(),
        total=len(sweet_vaults),
    ):
        diff_distri = feature_difference_distri[id]
        vaults = sweet_vaults[id]
        reuses = sweet_reuses[id]
        all_boosted_ngrams = load_pickle(
            f"{nle_dir}/model/{args.num}/boosted_ngrams_{id}.pkl"
        )
        score_list = []

        for i in range(args.num):
            boosted_ngrams = {}
            for prev, next_item in all_boosted_ngrams[i]:
                if prev not in boosted_ngrams:
                    boosted_ngrams[prev] = []
                boosted_ngrams[prev].append(next_item)

            vault = vaults[i]
            reuse = reuses[i]

            # single password
            Pr_decoy = []
            f_a = []
            for j, pw in enumerate(vault):
                pr_real = D.get(pw, 0)
                base_idx = nle.encoder.reuse_check(j, reuse)
                if base_idx == None:
                    bw = None
                else:
                    bw = vault[base_idx]
                pr_decoy = nle.adaptive_prob(bw, pw, boosted_ngrams, args.alpha)
                f_a.append(pr_real)
                Pr_decoy.append(pr_decoy)

            score = adaptive_hybrid(
                f_a,
                Pr_decoy,
                D_size,
                diff_distri[i],
                real_distribution,
                decoy_distribution,
                vault,
                boosted_ngrams,
            )

            if i == 0:
                score_list.append((True, score))
            else:
                score_list.append((False, score))

        random.shuffle(score_list)
        score_list = sorted(score_list, key=lambda x: x[1], reverse=True)
        index = next((k for k, t in enumerate(score_list) if t[0] == True), None)
        ranks[id] = index

    result_dir = os.path.join(nle_dir, f"results")
    os.makedirs(result_dir, exist_ok=True)
    output_path = f"{result_dir}/adaptive_hybrid_ranks_{args.num}.json"
    save_json(ranks, output_path)


if __name__ == "__main__":
    args = parse_command()
    main(args)

# python adaptive_hybrid.py --num 100
