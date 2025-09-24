import os
import sys
import json
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


linestyle_dict = {
    "solid": "solid",
    "dotted": "dotted",
    "dashed": "dashed",
    "dashdot": "dashdot",
    "loosely dotted": (0, (1, 10)),
    "densely dotted": (0, (1, 1)),
    "densely dashed": (0, (5, 1)),
    "dashed": (0, (5, 5)),
    "loosely dashed": (0, (5, 10)),
    "dashdotdotted": (0, (3, 5, 1, 5, 1, 5)),
    "densely dashdotted": (0, (3, 1, 1, 1)),
    "densely dashdotdotted": (0, (3, 1, 1, 1, 1, 1)),
    "loosely dashdotted": (0, (3, 10, 1, 10)),
    "loosely dashdotdotted": (0, (3, 10, 1, 10, 1, 10)),
    "dashdotted": (0, (3, 5, 1, 5)),
}

colors_dict = {
    "#74B22A": "#74B22A",
    "#E85542": "#E85542",
    "#597EE5": "#597EE5",
    "#F0782C": "#F0782C",
    "#9F47B0": "#9F47B0",
    "#F79B15": "#F79B15",
    "#D36B06": "#D36B06",
    "#19A2DA": "#19A2DA",
}


def load_data(filepath):
    if not os.path.exists(filepath):
        scheme = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
        print(
            f"File not found: {filepath}. Please return Step (2) to perform the corresponding distinguishing attack against {scheme}."
        )
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def point(r_freq, n):
    rank = []
    cdf = []
    p = 0
    total = sum(r_freq.values())
    for i in range(n):
        p += r_freq[i]
        rank.append(i / (n - 1))
        cdf.append(p / total)
    if n >= 100:
        indices = sorted(set([0, n - 1] + random.sample(range(1, n - 1), 98)))
        random_rank = [rank[i] for i in indices]
        random_cdf = [cdf[i] for i in indices]
        return random_rank, random_cdf
    else:
        return rank, cdf


def plot_cdfg(
    labels,
    linestyles,
    colors,
    files,
    num,
    k=None,
    writer=None,
    outfile=None,
    attack=None,
):

    alpha = 9
    width_cm = 4.00
    height_cm = 2.8
    figsize = ((width_cm / 2.54) * alpha, (height_cm / 2.54) * alpha)
    plt.figure(figsize=figsize)

    for linestyle, color, label, file in zip(linestyles, colors, labels, files):
        if file == None:
            continue
        data = load_data(f"{file}")
        if data == None:
            continue
        average_rank = (sum(data.values()) / len(data.keys())) / num
        writer.write(f"{label} average rank: {average_rank} {attack}\n")
        print(f"{label} average rank {average_rank} {attack}")
        r_freq = Counter(data.values())
        x, y = point(r_freq, num)
        plt.plot(x, y, color=color, linestyle=linestyle, label=label, linewidth=7)

    x = np.linspace(0, 1, num)
    cdf = (x - 0) / (1 - 0)
    plt.plot(x, cdf, color="black", linestyle="solid", label="Baseline", linewidth=7)

    plt.xscale("linear")
    plt.yscale("linear")
    plt.xlabel("Rank", size=36)
    plt.ylabel("Fraction of successfully cracked", size=36)
    plt.tick_params(axis="both", labelsize=34, pad=5)
    plt.legend(
        loc="best",
        fontsize=36,
        frameon=False,
        framealpha=0.9,
    )

    ax = plt.gca()
    ax.xaxis.set_tick_params(direction="in", which="both")
    ax.yaxis.set_tick_params(direction="in", which="both")
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")

    if outfile is not None:
        plt.savefig(f"{outfile}", dpi=300, bbox_inches="tight")

    # plt.show()


def parse_command():
    parser = argparse.ArgumentParser(description="Plot CDG figure.")
    parser.add_argument(
        "--scheme",
        required=False,
        type=str,
        default="nocrack",
    )
    parser.add_argument(
        "--num",
        required=False,
        type=int,
        default=100,
    )
    return parser.parse_args()


attacks = [
    "kl_divergence",
    "single_password",
    "theoretically_grounded",
    "weak_encoding",
    "strong_encoding",
    "password_similarity",
    "(adaptive)hybrid",
    "adaptive_extra",
]


def main():
    args = parse_command()

    labels = [
        "NoCrack-NLE",
        "VaultGuard",
    ]
    linestyles = [
        linestyle_dict["densely dotted"],
        linestyle_dict["densely dashed"],
    ]
    colors = [
        colors_dict["#E85542"],
        colors_dict["#74B22A"],
    ]
    n = int(args.num)

    if args.scheme == "nocrack":

        writer = open(f"{curr_dir}/results/nocrack_avg_rank.txt", "w")
        for attack in attacks:
            if attack == "adaptive_extra":
                continue

            output_file = ""

            if attack == "kl_divergence":
                files = [
                    f"{curr_dir}/results/kl_divergence_ranks_{n}.json",
                ]
                output_file = f"{curr_dir}/results/kl_divergence.pdf"

            if attack == "weak_encoding":
                files = [
                    f"{curr_dir}/results/weak_encoding_ranks_{n}.json",
                ]
                output_file = f"{curr_dir}/results/weak_encoding.pdf"

            if attack == "strong_encoding":
                files = [
                    f"{curr_dir}/results/strong_encoding_ranks_{n}.json",
                ]
                output_file = f"{curr_dir}/results/strong_encoding.pdf"

            if attack == "single_password":
                files = [
                    f"{curr_dir}/results/single_password_ranks_{n}.json",
                ]
                output_file = f"{curr_dir}/results/single_password.pdf"

            if attack == "password_similarity":
                files = [
                    f"{curr_dir}/results/password_similarity_ranks_{n}.json",
                ]
                output_file = f"{curr_dir}/results/password_similarity.pdf"

            if attack == "(adaptive)hybrid":
                files = [
                    f"{curr_dir}/results/hybrid_ranks_{n}.json",
                ]
                output_file = f"{curr_dir}/results/adaptive_hybrid.pdf"

            if attack == "theoretically_grounded":
                files = [
                    f"{curr_dir}/results/theoretically_grounded_ranks_{n}.json",
                ]
                output_file = f"{curr_dir}/results/theoretically_grounded.pdf"

            plot_cdfg(
                labels,
                linestyles,
                colors,
                files,
                num=n,
                k=None,
                writer=writer,
                outfile=output_file,
                attack=attack,
            )
        writer.close()

    if args.scheme == "both":

        writer = open(f"{curr_dir}/results/both_avg_rank.txt", "w")
        for attack in attacks:

            output_file = ""

            if attack == "kl_divergence":
                files = [
                    f"{curr_dir}/results/kl_divergence_ranks_{n}.json",
                    f"{root_dir}/VaultGuard/results/kl_divergence_ranks_{n}.json",
                ]
                output_file = f"{curr_dir}/results/kl_divergence.pdf"

            if attack == "weak_encoding":
                files = [
                    f"{curr_dir}/results/weak_encoding_ranks_{n}.json",
                    f"{root_dir}/VaultGuard/results/weak_encoding_ranks_{n}.json",
                ]
                output_file = f"{curr_dir}/results/weak_encoding.pdf"

            if attack == "strong_encoding":
                files = [
                    f"{curr_dir}/results/strong_encoding_ranks_{n}.json",
                    f"{root_dir}/VaultGuard/results/strong_encoding_ranks_{n}.json",
                ]
                output_file = f"{curr_dir}/results/strong_encoding.pdf"

            if attack == "single_password":
                files = [
                    f"{curr_dir}/results/single_password_ranks_{n}.json",
                    f"{root_dir}/VaultGuard/results/single_password_ranks_{n}.json",
                ]
                output_file = f"{curr_dir}/results/single_password.pdf"

            if attack == "password_similarity":
                files = [
                    f"{curr_dir}/results/password_similarity_ranks_{n}.json",
                    f"{root_dir}/VaultGuard/results/password_similarity_ranks_{n}.json",
                ]
                output_file = f"{curr_dir}/results/password_similarity.pdf"

            if attack == "(adaptive)hybrid":
                files = [
                    f"{curr_dir}/results/hybrid_ranks_{n}.json",
                    f"{root_dir}/VaultGuard/results/adaptive_hybrid_ranks_{n}.json",
                ]
                output_file = f"{curr_dir}/results/adaptive_hybrid.pdf"

            if attack == "theoretically_grounded":
                files = [
                    f"{curr_dir}/results/theoretically_grounded_ranks_{n}.json",
                    f"{root_dir}/VaultGuard/results/theoretically_grounded_ranks_{n}.json",
                ]
                output_file = f"{curr_dir}/results/theoretically_grounded.pdf"

            if attack == "password_similarity":
                files = [
                    f"{curr_dir}/results/password_similarity_ranks_{n}.json",
                    f"{root_dir}/VaultGuard/results/password_similarity_ranks_{n}.json",
                ]
                output_file = f"{curr_dir}/results/password_similarity.pdf"

            if attack == "adaptive_extra":
                files = [
                    None,
                    f"{root_dir}/VaultGuard/results/adaptive_extra_ranks_{n}.json",
                ]
                output_file = f"{curr_dir}/results/adaptive_extra.pdf"

            plot_cdfg(
                labels,
                linestyles,
                colors,
                files,
                num=n,
                k=None,
                writer=writer,
                outfile=output_file,
                attack=attack,
            )
        writer.close()


if __name__ == "__main__":
    main()

# python plot.py --scheme nocrack --num 100
