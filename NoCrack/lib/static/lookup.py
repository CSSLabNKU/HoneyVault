import  os
import gzip
import bz2
import json
from dawg import IntDAWG
from collections import OrderedDict

# returns the type of file.
def file_type(filename, param='rb'):
    magic_dict = {
        b"\x1f\x8b\x08": "gz",
        b"\x42\x5a\x68": "bz2",
        b"\x50\x4b\x03\x04": "zip"
    }
    if param.startswith('w'):
        return filename.split('.')[-1]
    max_len = max(len(x) for x in magic_dict)
    with open(filename, 'rb') as f:
        file_start = f.read(max_len)
    for magic, filetype in list(magic_dict.items()):
        if file_start.startswith(magic):
            return filetype
    return "no match"


def open_(filename, mode='r'):
    type_ = file_type(filename, mode)
    if type_ == "bz2":
        f = bz2.open(filename, mode)
    elif type_ == "gz":
        f = gzip.open(filename, mode)
    else:
        f = open(filename, mode)
    return f

def load_dawg(f, t=IntDAWG):
    T = t()
    T.read(open_(f))
    return T

def save_dawg(T, fname):
    if not fname.endswith('gz'):
        fname = fname + '.gz'
    with gzip.open(fname, 'w') as f:
        T.write(f)

curr_file_path = os.path.abspath(__file__)
curr_dir = os.path.dirname(curr_file_path)
parent_dir = os.path.dirname(curr_dir)

# filename = input("Filename: ")
filename = 'grammar.cfg.gz'
filepath = f'{curr_dir}/{filename}'
content = open_(filepath, 'r')
grammar = json.load(content, object_pairs_hook=OrderedDict)

# Create dawg of the Wlist items for fast retrieval
Wlist = [(x, f) for k, v in list(grammar.items()) for x, f in v.items() if k.startswith('W')]
Wdawg = IntDAWG(Wlist)

# print(Wdawg['ambiguous'])

save_dawg(Wdawg, f'{parent_dir}/data/{filename}.dawg')

with open(f'{curr_dir}/{filename}.json', 'w', encoding='utf-8') as f:
    json.dump(grammar, f, indent=4)