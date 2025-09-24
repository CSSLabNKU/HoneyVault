import json
import pickle
from collections import defaultdict, Counter
from textwrap import indent


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_pickle(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def save_txt(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(f"{item}\n")


def load_txt(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = []
        for line in f:
            pw1, pw2, freq = line.strip().split("\t")
            data.append((pw1, pw2, int(freq)))
    return data


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, file_path, indent=4):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def pdf2cdf(pdf):
    cdf = [pdf[0]]
    for i in range(1, len(pdf)):
        cdf.append(cdf[i - 1] + pdf[i])
    return [i / cdf[-1] for i in cdf]


def nested_defaultdict():
    return defaultdict(Counter)
