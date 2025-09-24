import pickle
import json


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def save_json(data, filename, indent=4):
    with open(filename, "w") as f:
        json.dump(data, f, indent=indent)


def load_txt(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def pdf2cdf(pdf):
    cdf = [pdf[0]]
    for i in range(1, len(pdf)):
        cdf.append(cdf[-1] + pdf[i])
    return cdf
