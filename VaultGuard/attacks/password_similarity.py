"""
Password similarity attack against VaultGuard NLE.

For VaultGuard-NLE: M = "LCSStr", I = "" (["LCS", "Manhattan", "Overlap", "Levenshtein"])
"""

import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
nle_dir = os.path.dirname(curr_dir)
sys.path.append(nle_dir)

import random
import argparse
from tqdm import tqdm
import Levenshtein
from difflib import SequenceMatcher
from collections import Counter
from toolkits import load_pickle, save_pickle, save_json


def LCSStr(s1, s2):
    """
    The password pair has Feature LCSStr if the length of their longest common substring is at least half of the maximum length of them.
    """
    matcher = SequenceMatcher(None, s1, s2)
    longest_match = matcher.find_longest_match(0, len(s1), 0, len(s2))
    lcsstr = s1[longest_match.a : longest_match.a + longest_match.size]
    max_len = max(len(s1), len(s2))
    similarity = len(lcsstr) / max_len
    return similarity >= 0.5


def levenshtein(s1, s2):
    """
    Levenshtein determine the number of edit operations (insertion, deletion, replacement, transposition) required to transform one string into another. For each meter F, we define that (pw1, pw2) has Feature F, if the similarity score of (pw1, pw2) is at least 0.5 under the meter F.
    """
    distance = Levenshtein.distance(s1, s2)
    max_len = max(len(s1), len(s2))
    similarity = 1 - distance / max_len
    return similarity >= 0.5


def LCS(s1, s2):
    """
    If the length of the longest common subsequence of the two passwords is at least half of the maximum length of them.
    For each metric F, we define that (pw1, pw2) has Feature F if the similarity score of (pw1, pw2) is at least 0.5 under metric F.
    A subsequence of a string is a new string formed by deleting some (or no) characters from the original string without changing the relative order of the remaining characters.
    For example, "ace" is a subsequence of "abcde", but "aec" is not a subsequence of "abcde".
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]
    max_len = max(m, n)
    similarity = lcs_len / max_len
    return similarity >= 0.5


def Manhattan(s1, s2):
    """
    The Manhattan distance of the two passwords is at least half of  the length sum of them.
    """
    s1_counter = Counter(s1)
    s2_counter = Counter(s2)
    common_chars = set(s1 + s2)
    manhattan_distance = sum(
        abs(s1_counter[char] - s2_counter[char]) for char in common_chars
    )
    similarity = 1 - (manhattan_distance / (len(s1) + len(s2)))
    return similarity >= 0.5


def Overlap(s1, s2):
    """
    The union size of the character sets of the two passwords is at  least half of the minimal size of the character sets of them.
    """
    set1 = set(s1)
    set2 = set(s2)
    overlap = len(set1 & set2)
    max_len = min(len(s1), len(s2))
    similarity = overlap / max_len
    return similarity >= 0.5


def is_vault_has_Feature(vault, F1, F2):
    for i in range(len(vault)):
        pw1 = vault[i]
        for j in range(i + 1, len(vault)):
            pw2 = vault[j]
            if F1(pw1, pw2) and not F2(pw1, pw2):
                return True
    return False


def feature_difference_distribution(M, I, sweetvaults):
    output_dir = os.path.join(nle_dir, "results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    feature_difference_distri = {}

    for id, vaults in tqdm(sweetvaults.items(), total=len(sweetvaults)):
        feature_difference_distri[id] = []

        for vault in vaults:
            M_I = is_vault_has_Feature(vault, M, I)
            I_M = is_vault_has_Feature(vault, I, M)

            feature_difference_distri[id].append((M_I, I_M))

    save_pickle(
        feature_difference_distri,
        os.path.join(output_dir, f"{M.__name__}_and_{I.__name__}.pkl"),
    )


def password_similarity(x_list, real_distri, decoy_distri):
    """USENIX21: Password similarity attack
    x_list: List of feature difference values, e.g., [0, 1], representing the values of M|I and I|M
    real_distri: Feature difference distribution of the real password vault, e.g., [real_M_I, real_I_M]
    decoy_distri: Feature difference distribution of the decoy password vault, e.g., [decoy_M_I, decoy_I_M]
    """
    score = (real_distri["M_I"][x_list[0]] / real_distri["M_I"]["_total_"]) / (
        decoy_distri["M_I"][x_list[0]] / decoy_distri["M_I"]["_total_"]
    )
    score *= (real_distri["I_M"][x_list[1]] / real_distri["I_M"]["_total_"]) / (
        decoy_distri["I_M"][x_list[1]] / decoy_distri["I_M"]["_total_"]
    )
    return score


feature_functions = {
    "LCSStr": LCSStr,
    "levenshtein": levenshtein,
    "LCS": LCS,
    "Manhattan": Manhattan,
    "Overlap": Overlap,
}


def parse_command():
    parser = argparse.ArgumentParser(description="Password similarity attack")
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
    return parser.parse_args()


def main(args):

    sweet_vaults = load_pickle(f"{nle_dir}/results/sweetvaults_{args.num}.pkl")

    feature_dir = os.path.join(nle_dir, "results")
    feature_difference_file = os.path.join(feature_dir, f"{args.M}_and_{args.I}.pkl")
    if not os.path.exists(feature_difference_file):
        feature_difference_distribution(
            feature_functions[args.M],
            feature_functions[args.I],
            sweet_vaults,
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
    for id, diff_distri in tqdm(
        feature_difference_distri.items(), total=len(feature_difference_distri)
    ):
        score_list = []

        for i in range(args.num):
            if i == 0:
                score = password_similarity(
                    diff_distri[i], real_distribution, decoy_distribution
                )
                score_list.append((True, score))
            else:
                score = password_similarity(
                    diff_distri[i], real_distribution, decoy_distribution
                )
                score_list.append((False, score))

        random.shuffle(score_list)
        score_list = sorted(score_list, key=lambda x: x[1], reverse=True)
        index = next((k for k, t in enumerate(score_list) if t[0] == True), None)
        ranks[id] = index

    result_dir = os.path.join(nle_dir, "results")
    os.makedirs(result_dir, exist_ok=True)
    save_json(ranks, f"{result_dir}/password_similarity_ranks_{args.num}.json", 4)


if __name__ == "__main__":
    args = parse_command()
    main(args)

# python password_similarity.py --num 1000
