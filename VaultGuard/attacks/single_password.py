"""
Single password attack against VaultGuard NLE.
"""

import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
nle_dir = os.path.dirname(curr_dir)
sys.path.append(nle_dir)

import random
import argparse
from tqdm import tqdm
from vaultguard_nle import init_vaultguard_nle
from toolkits import load_pickle, save_json


def single_password(f_a, Pr_decoy, D_size, a=1, f_d=5):
    """USENIX21: Single-password attack

    Score function
    P_opt = Π_pw P_sp(pw), P_sp(pw) = Pr_smooth(pw) / Pr_decoy(pw)
    if f_a(pw) <= 5 and P_sp(pw) > 1: P_sp(pw) = 1; else: P_sp(pw) remains unchanged

    Details:
    Pr_decoy: can be calculated by the single-password models in honey vault args.schemes
    f_a(pw) : the absolute frequency of pw in the training set D
    D_size  : the size of the training set D
    a = 1   : is a smoothing parameter
    f_d = 5 : is a parameter representing the demarcation line between high-frequency and low-frequency passwords
    """
    score = 1.0
    for fa, fd in zip(f_a, Pr_decoy):
        fr = (fa + a) / (D_size + a)
        p = fr / fd
        if fa <= f_d and p > 1:
            score *= 1
        else:
            score *= p
        score *= p
    return score


def parse_command():
    parser = argparse.ArgumentParser(description="Single password attack")
    parser.add_argument(
        "--num",
        required=False,
        type=int,
        default=100,
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
        default=2,
        required=False,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default="wishbone",
    )
    return parser.parse_args()


def main_v1(args):
    nle = init_vaultguard_nle()
    sweet_vaults = load_pickle(f"{nle_dir}/results/sweetvaults_{args.num}.pkl")
    sweet_reuses = load_pickle(f"{nle_dir}/results/sweetreuses_{args.num}.pkl")
    D = load_pickle(f"{nle_dir}/data/{args.dataset}_counter.pkl")
    D_size = sum(D.values())

    ranks = {}
    for id in tqdm(sweet_vaults.keys(), total=len(sweet_vaults)):
        all_boosted_ngrams = load_pickle(
            f"{nle_dir}/model/{args.num}/boosted_ngrams_{id}.pkl"
        )
        vaults = sweet_vaults[id]
        reuses = sweet_reuses[id]
        score_list = []

        for i in range(args.num):
            boosted_ngrams = {}
            for prev, next_item in all_boosted_ngrams[i]:
                if prev not in boosted_ngrams:
                    boosted_ngrams[prev] = []
                boosted_ngrams[prev].append(next_item)

            vault = vaults[i]
            reuse = reuses[i]
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

            score = single_password(f_a, Pr_decoy, D_size)

            if i == 0:
                score_list.append((True, score))
            else:
                score_list.append((False, score))

        index_list = []
        for _ in range(3):
            random.shuffle(score_list)
            score_list = sorted(score_list, key=lambda x: x[1], reverse=True)
            index = next((k for k, t in enumerate(score_list) if t[0] == True), None)
            index_list.append(index)
        ranks[id] = sum(index_list) // len(index_list)

    result_dir = os.path.join(nle_dir, f"results")
    os.makedirs(result_dir, exist_ok=True)
    output_path = f"{result_dir}/single_password_ranks_{args.num}.json"
    save_json(ranks, output_path)


def main_v2(args):
    nle = init_vaultguard_nle()
    sweet_vaults = load_pickle(f"{nle_dir}/results/sweetvaults_{args.num}.pkl")
    sweet_reuses = load_pickle(f"{nle_dir}/results/sweetreuses_{args.num}.pkl")
    D = load_pickle(f"{nle_dir}/data/{args.dataset}_counter.pkl")
    D_size = sum(D.values())

    ranks = {}
    for id in tqdm(sweet_vaults.keys(), total=len(sweet_vaults)):
        all_boosted_ngrams = load_pickle(
            f"{nle_dir}/model/{args.num}/boosted_ngrams_{id}.pkl"
        )
        vaults = sweet_vaults[id]
        reuses = sweet_reuses[id]
        score_list = []

        for i in range(args.num):
            boosted_ngrams = {}
            for prev, next_item in all_boosted_ngrams[i]:
                if prev not in boosted_ngrams:
                    boosted_ngrams[prev] = []
                boosted_ngrams[prev].append(next_item)

            vault = vaults[i]
            reuse = reuses[i]
            Pr_decoy = []
            f_a = []

            for j, pw in enumerate(vault):
                pr_real = D.get(pw, 0)
                base_idx = nle.encoder.reuse_check(j, reuse)
                if base_idx != None:
                    continue
                pr_decoy = nle.encoder.adaptive_sppm_prob(
                    pw, boosted_ngrams, args.alpha
                )
                f_a.append(pr_real)
                Pr_decoy.append(pr_decoy)

            score = single_password(f_a, Pr_decoy, D_size)

            if i == 0:
                score_list.append((True, score))
            else:
                score_list.append((False, score))

        index_list = []
        for _ in range(1):
            random.shuffle(score_list)
            score_list = sorted(score_list, key=lambda x: x[1], reverse=True)
            index = next((k for k, t in enumerate(score_list) if t[0] == True), None)
            index_list.append(index)
        ranks[id] = sum(index_list) // len(index_list)

    result_dir = os.path.join(nle_dir, f"results")
    os.makedirs(result_dir, exist_ok=True)
    output_path = f"{result_dir}/single_password_ranks_{args.num}.json"
    save_json(ranks, output_path)


if __name__ == "__main__":
    args = parse_command()
    main_v2(args)


# python single_password.py --num 100
