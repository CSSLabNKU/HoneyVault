import os
import gzip
import bz2
from dawg import IntDAWG

# returns the type of file.
def file_type(filename, param='rb'):
    magic_dict = {
        b"\x1f\x8b\x08": "gz",
        b"\x42\x5a\x68": "bz2",
        b"\x50\x4b\x03\x04": "zip"
    }
    if param.startswith('w'):
        # 写模式,文件类型直接由扩展名决定。
        return filename.split('.')[-1]
    max_len = max(len(x) for x in magic_dict)
    with open(filename, 'rb') as f:
        file_start = f.read(max_len)
    for magic, filetype in list(magic_dict.items()):
        # 匹配文件类型
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

# 获取当前文件路径
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

file_path = os.path.join(current_dir, 'grammar.cfg.gz.dawg.gz')

# 解析 .dawg 文件内容 --> 如何输出？
content = load_dawg(file_path)

print(content['ambiguous'])
