import os
import random
from math import log
from difflib import SequenceMatcher
from collections import defaultdict, Counter
from tqdm import tqdm
from toolkits import load_pickle, save_pickle, save_txt, nested_defaultdict, pdf2cdf


class IS_DTE:
    def __init__(self, l):
        self.MAX_INT = 2**l

    def padding(self, num):
        return [random.randint(0, self.MAX_INT - 1) for _ in range(num)]

    def rep(self, a):
        return round(a * self.MAX_INT)

    def encode(self, m, c):
        cdf = c.copy()
        cdf.insert(0, 0)
        m = m + 1
        left = self.rep(cdf[m - 1])
        right = self.rep(cdf[m]) - 1
        s = random.randint(left, right)
        return s

    def decode(self, s, c):
        cdf = c.copy()
        cdf.insert(0, 0)
        for m in range(len(cdf)):
            if self.rep(cdf[m]) > s:
                return m - 1


def HEAdap(vaults, args):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ngrams_distribution = load_pickle(f"{curr_dir}/model/ngrams_distribution_4.pkl")
    prev = " " * 4
    chars = list(ngrams_distribution[prev].keys())
    weights = list(ngrams_distribution[prev].values())

    for id in vaults.keys():
        vault = vaults[id]
        boosted_ngrams = {i: [] for i in range(args.num)}
        boosted_ngrams[0] = [(prev, pw[0]) for pw in vault]
        size_counter = Counter(boosted_ngrams[0])

        for i in range(1, args.num):
            nexts = [
                c for c in random.choices(chars, weights=weights, k=len(size_counter))
            ]
            for next, size in zip(nexts, size_counter.values()):
                boosted_ngrams[i].extend([(prev, next)] * size)

        result_dir = f"{curr_dir}/model/{args.num}"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        save_pickle(boosted_ngrams, f"{result_dir}/boosted_ngrams_{id}.pkl")


class Encoder:
    def __init__(self, order, pseudocount, dte):
        self.order = order
        self.pseudocount = pseudocount
        self.dte = dte
        self.start = " "
        self.l_min = 4
        self.l_max = 32
        self.max_code_len = 100

    def sppm_parse(self, pw):
        length = len(pw)
        if length < self.l_min or length > self.l_max:
            raise ValueError("Invalid password length")
        rules = [length]
        pw = self.start * self.order + pw
        for i in range(self.order, len(pw)):
            prev = pw[i - self.order : i]
            next = pw[i]
            rules.append((prev, next))
        return rules

    def sppm_generate(self, rules):
        return "".join([rule[1] for rule in rules[1:]])

    def sppm_sample(self, num=1):
        sample_list = []
        char_lst = list(self.charset)
        sqrt_num = int(num**0.5) + 1
        len_lst = list(self.length_counts.keys())
        l_lst = random.choices(
            len_lst, weights=list(self.length_counts.values()), k=sqrt_num
        )
        for l in l_lst:
            for _ in range(sqrt_num):
                pw = self.start * self.order
                for i in range(self.order, l + self.order):
                    prev = pw[i - self.order : i]
                    if prev not in self.sppm[str(l)]:
                        next_char = random.choice(char_lst)
                    else:
                        weights = [
                            self.sppm[str(l)][prev].get(c, 0) + self.pseudocount
                            for c in char_lst
                        ]
                        next_char = random.choices(char_lst, weights=weights, k=1)[0]
                    pw += next_char
                sample_list.append(pw[self.order :])
        return sample_list[:num]

    def sppm_prob(self, pw):
        prob = 1.0
        rules = self.sppm_parse(pw)
        l = rules.pop(0)
        if l < self.l_min or l > self.l_max:
            raise ValueError("Invalid password length")
        else:
            DISTRI_NORM_FACTOR = self.length_counts[str(l)] / sum(
                self.length_counts.values()
            )
        for prev, next in rules:
            if prev not in self.sppm[str(l)]:
                prob *= self.uniform_char_prob
            else:
                smoothed_count = (
                    self.sppm[str(l)][prev].get(next, 0.0) + self.pseudocount
                )
                total_count = sum(
                    self.sppm[str(l)][prev].values()
                ) + self.pseudocount * len(self.charset)
                prob *= smoothed_count / total_count
        return prob * DISTRI_NORM_FACTOR

    def adaptive_sppm_prob(self, pw, boosted_ngrams, alpha=2):
        prob = 1.0
        rules = self.sppm_parse(pw)
        l = rules.pop(0)
        for prev, next in rules:
            cnt = 0
            prev_next_table = {}
            if prev in self.sppm[str(l)]:
                prev_next_table = dict(self.sppm[str(l)][prev])
            for c in prev_next_table.keys():
                prev_next_table[c] += self.pseudocount
            if prev in boosted_ngrams:
                for c in boosted_ngrams[prev]:
                    if c in prev_next_table:
                        cnt += 1
                        prev_next_table[c] *= alpha
                    else:
                        cnt += 1
                        prev_next_table[c] = self.pseudocount * alpha

            if len(prev_next_table) == 0:
                prob *= self.uniform_char_prob
            else:
                smoothed_count = prev_next_table.get(next, self.pseudocount)
                total_count = sum(
                    [prev_next_table.get(c, self.pseudocount) for c in self.charset]
                )
                prob *= smoothed_count / total_count
        if l < self.l_min or l > self.l_max:
            raise ValueError("Invalid password length")
        else:
            DISTRI_NORM_FACTOR = (self.length_counts[str(l)] * (alpha**cnt)) / (
                sum(self.length_counts.values())
                + self.length_counts[str(l)] * (alpha**cnt - 1)
            )
        return prob * DISTRI_NORM_FACTOR

    def sppm_encode(self, pw):
        code = []
        rules = self.sppm_parse(pw)
        pw_len = rules.pop(0)
        len_idx = self.len2idx[pw_len]
        code.append(self.dte.encode(len_idx, self.length_cdf))
        char_lst = list(self.charset)
        for prev, next in rules:
            next_idx = self.char2idx[next]
            if prev not in self.sppm[str(pw_len)]:
                code.append(self.dte.encode(next_idx, self.uniform_char_cdf))
            else:
                char_pdf = [
                    self.sppm[str(pw_len)][prev].get(c, 0) + self.pseudocount
                    for c in char_lst
                ]
                code.append(self.dte.encode(next_idx, pdf2cdf(char_pdf)))
        if len(code) < self.max_code_len:
            PADDING_LENGTH = self.max_code_len - len(code)
            code.extend(self.dte.padding(PADDING_LENGTH))
        return code

    def sppm_decode(self, code):
        len_idx = self.dte.decode(code[0], self.length_cdf)
        pw_len = self.idx2len[len_idx]
        char_lst = list(self.charset)
        pw = self.start * self.order
        for i in range(pw_len):
            prev = pw[i : i + self.order]
            if prev not in self.sppm[str(pw_len)]:
                next_idx = self.dte.decode(code[i + 1], self.uniform_char_cdf)
            else:
                char_pdf = [
                    self.sppm[str(pw_len)][prev].get(c, 0) + self.pseudocount
                    for c in char_lst
                ]
                next_idx = self.dte.decode(code[i + 1], pdf2cdf(char_pdf))
            pw += self.charset[next_idx]
        return pw[self.order :]

    def adaptive_sppm_encode(self, pw, boosted_ngrams, alpha=2):
        code = []
        rules = self.sppm_parse(pw)
        pw_len = rules.pop(0)
        len_idx = self.len2idx[pw_len]
        code.append(self.dte.encode(len_idx, self.length_cdf))
        char_lst = list(self.charset)
        for prev, next in rules:
            next_idx = self.char2idx[next]

            prev_next_table = {}
            if prev in self.sppm[str(pw_len)]:
                prev_next_table = dict(self.sppm[str(pw_len)][prev])
            for c in prev_next_table.keys():
                prev_next_table[c] += self.pseudocount
            if prev in boosted_ngrams:
                for c in boosted_ngrams[prev]:
                    if c in prev_next_table:
                        prev_next_table[c] *= alpha
                    else:
                        prev_next_table[c] = self.pseudocount * alpha

            if len(prev_next_table) == 0:
                code.append(self.dte.encode(next_idx, self.uniform_char_cdf))
            else:
                next_pdf = [prev_next_table.get(c, self.pseudocount) for c in char_lst]
                code.append(self.dte.encode(next_idx, pdf2cdf(next_pdf)))
        if len(code) < self.max_code_len:
            PADDING_LENGTH = self.max_code_len - len(code)
            code.extend(self.dte.padding(PADDING_LENGTH))
        return code

    def adaptive_sppm_decode(self, code, boosted_ngrams, alpha=2):
        len_idx = self.dte.decode(code.pop(), self.length_cdf)
        pw_len = self.idx2len[len_idx]
        char_lst = list(self.charset)
        pw = self.start * self.order
        for i in range(pw_len):
            prev = pw[i : i + self.order]

            prev_next_table = {}
            if prev in self.sppm[str(pw_len)]:
                prev_next_table = dict(self.sppm[str(pw_len)][prev])
            for c in prev_next_table.keys():
                prev_next_table[c] += self.pseudocount
            if prev in boosted_ngrams:
                for c in boosted_ngrams[prev]:
                    if c in prev_next_table:
                        prev_next_table[c] *= alpha
                    else:
                        prev_next_table[c] = self.pseudocount * alpha

            if len(prev_next_table) == 0:
                next_idx = self.dte.decode(code[i], self.uniform_char_cdf)
            else:
                next_pdf = [prev_next_table.get(c, self.pseudocount) for c in char_lst]
                next_idx = self.dte.decode(code[i], pdf2cdf(next_pdf))
            pw += self.charset[next_idx]
        return pw[self.order :]

    def adaptive_sppm_decode_path(self, code, boosted_ngrams, alpha=2):
        len_idx = self.dte.decode(code.pop(), self.length_cdf)
        pw_len = self.idx2len[len_idx]
        char_lst = list(self.charset)
        pw = self.start * self.order
        rules = []
        for i in range(pw_len):
            prev = pw[i : i + self.order]

            prev_next_table = {}
            if prev in self.sppm[str(pw_len)]:
                prev_next_table = dict(self.sppm[str(pw_len)][prev])
            for c in prev_next_table.keys():
                prev_next_table[c] += self.pseudocount
            if prev in boosted_ngrams:
                for c in boosted_ngrams[prev]:
                    if c in prev_next_table:
                        prev_next_table[c] *= alpha
                    else:
                        prev_next_table[c] = self.pseudocount * alpha

            if len(prev_next_table) == 0:
                next_idx = self.dte.decode(code[i], self.uniform_char_cdf)
            else:
                next_pdf = [prev_next_table.get(c, self.pseudocount) for c in char_lst]
                next_idx = self.dte.decode(code[i], pdf2cdf(next_pdf))
            pw += self.charset[next_idx]
            rules.append((prev, self.charset[next_idx]))
        return rules

    def sppm_train(self, data, save_dir):
        length_counts = defaultdict(int)
        sppm = defaultdict(nested_defaultdict)

        groups = defaultdict(lambda: defaultdict(int))
        if not isinstance(data, dict):
            if isinstance(data, list):
                data = Counter(data)
            else:
                raise ValueError("Training set must be a dictionary or list")
        for pw, freq in tqdm(data.items(), total=len(data), desc="Grouping by len"):
            l = len(pw)
            if self.l_min <= l <= self.l_max:
                groups[l][pw] = freq
                length_counts[l] += freq
        print(f"Length: {list(length_counts.keys())}")

        for l, item in tqdm(groups.items(), total=len(groups), desc="Training sppm"):
            for pw, freq in item.items():
                rules = self.sppm_parse(pw)
                for rule in rules[1:]:
                    prev, next = rule
                    sppm[str(l)][prev][next] += freq

        print("Saving sppm model.")
        os.makedirs(save_dir, exist_ok=True)
        save_pickle(self.charset, f"{save_dir}/charset.pkl")
        save_pickle(sppm, f"{save_dir}/markov_{self.order}.pkl")
        save_pickle(length_counts, f"{save_dir}/length_counts.pkl")

    def load_sppm_model(self, model_dir):
        self.charset = load_pickle(f"{model_dir}/charset.pkl")
        self.sppm = load_pickle(f"{model_dir}/markov_{self.order}.pkl")
        self.length_counts = load_pickle(f"{model_dir}/length_counts.pkl")
        self.char2idx = {char: idx for idx, char in enumerate(self.charset)}

        self.uniform_char_prob = 1.0 / len(self.charset)
        self.uniform_char_cdf = pdf2cdf([1] * len(self.charset))
        self.length_cdf = pdf2cdf(list(self.length_counts.values()))
        self.len2idx = {int(l): i for i, l in enumerate(self.length_counts.keys())}
        self.idx2len = {i: int(l) for i, l in enumerate(self.length_counts.keys())}

    def LCSStr(self, s1, s2):
        matcher = SequenceMatcher(None, s1, s2)
        longest_match = matcher.find_longest_match(0, len(s1), 0, len(s2))
        lcsstr = s1[longest_match.a : longest_match.a + longest_match.size]
        return lcsstr

    def find_lcsstr_pairs(self, D, output_file):
        if not isinstance(D, dict):
            if isinstance(D, list):
                D = Counter(D)
            else:
                raise ValueError("Training set must be a dictionary or list")
        passwords = list(D.keys())
        n = len(passwords)
        pairs = []
        for i in tqdm(range(n), desc="Finding LCSStr pairs"):
            for j in range(i + 1, n):
                pw1 = passwords[i]
                pw2 = passwords[j]
                max_len = max(len(pw1), len(pw2))
                if len(self.LCSStr(pw1, pw2)) >= (max_len / 2):
                    freq = min(D[pw1], D[pw2])
                    pairs.append(f"{pw1}\t{pw2}\t{freq}")
        save_txt(pairs, output_file)

    def is_reuse(self, bw, pw):
        lcs = self.LCSStr(bw, pw)
        max_len = max(len(bw), len(pw))
        return len(lcs) >= (max_len / 2)

    def multi_reuse(self, vault):
        excluded_idx = set()
        reused = []
        for idx in range(len(vault)):
            if idx in excluded_idx:
                continue

            excluded_idx.add(idx)
            bw = vault[idx]

            for candidate in range(idx + 1, len(vault)):
                if candidate in excluded_idx:
                    continue

                pw = vault[candidate]
                if self.is_reuse(bw, pw):
                    excluded_idx.add(candidate)
                    reused.append([candidate, idx])

        reused.sort(key=lambda x: x[0])
        return reused  # [[reused_idx, base_idx], ...]

    def reuse_check(self, index, reused):
        for reused_idx, base_idx in reused:
            if index == reused_idx:
                return base_idx  # return the index of bw of the reused pw
        return None  # there is no reuse

    def prpm_parse(self, bw, pw):
        if not self.is_reuse(bw, pw):
            return None

        lcsstr = self.LCSStr(bw, pw)
        bw_start = bw.find(lcsstr)
        pw_start = pw.find(lcsstr)
        bw_end = bw_start + len(lcsstr)
        pw_end = pw_start + len(lcsstr)

        rules = {"H": None, "T": None}

        h_del_size = bw_start
        h_add_char = pw[:pw_start]
        h_add_size = len(h_add_char)
        head_ops = {"MOD": "None", "HDN": 0, "HAN": 0, "HAC": []}
        if h_del_size > 0 and h_add_size > 0:
            head_ops["MOD"] = "Delete-Add"
            head_ops["HDN"] = h_del_size
            head_ops["HAN"] = h_add_size
            head_ops["HAC"] = list(h_add_char)
        elif h_del_size > 0 and h_add_size == 0:
            head_ops["MOD"] = "Delete"
            head_ops["HDN"] = h_del_size
        elif h_del_size == 0 and h_add_size > 0:
            head_ops["MOD"] = "Add"
            head_ops["HAN"] = h_add_size
            head_ops["HAC"] = list(h_add_char)
        rules["H"] = head_ops

        t_del_size = len(bw) - bw_end
        t_add_char = pw[pw_end:]
        t_add_size = len(t_add_char)
        tail_ops = {"MOD": "None", "TDN": 0, "TAN": 0, "TAC": []}
        if t_del_size > 0 and t_add_size > 0:
            tail_ops["MOD"] = "Delete-Add"
            tail_ops["TDN"] = t_del_size
            tail_ops["TAN"] = t_add_size
            tail_ops["TAC"] = list(t_add_char)
        elif t_del_size > 0 and t_add_size == 0:
            tail_ops["MOD"] = "Delete"
            tail_ops["TDN"] = t_del_size
        elif t_del_size == 0 and t_add_size > 0:
            tail_ops["MOD"] = "Add"
            tail_ops["TAN"] = t_add_size
            tail_ops["TAC"] = list(t_add_char)
        rules["T"] = tail_ops

        return rules

    def prpm_generate(self, bw, rules):
        head_ops = rules["H"]
        tail_ops = rules["T"]
        pw = self.modify_head(bw, head_ops)
        pw = self.modify_tail(pw, tail_ops)
        return pw

    def prpm_prob(self, bw, pw, boosted_ngrams=None):
        rules = self.prpm_parse(bw, pw)
        if not rules:
            print(bw, pw)
            return None

        head_ops = rules["H"]
        tail_ops = rules["T"]

        bw_len = len(bw)
        if boosted_ngrams == None:
            prob = self.sppm_prob(bw)
        else:
            prob = self.adaptive_sppm_prob(bw, boosted_ngrams)

        hdn = head_ops.get("HDN", 0)
        MAX_HDN = bw_len // 2  # l_HD <= 1/2 * l_old
        valid_HDN = self.get_valid_distribution(self.prpm["HDN"], MAX_HDN)
        if hdn != 0:
            prob *= valid_HDN[hdn] / sum(valid_HDN.values())

        tdn = tail_ops.get("TDN", 0)
        MAX_TDN = bw_len // 2 - hdn  # l_TD <= 1/2 * l_old - l_HD
        valid_TDN = self.get_valid_distribution(self.prpm["TDN"], MAX_TDN)
        if tdn != 0:
            prob *= valid_TDN[tdn] / sum(valid_TDN.values())

        han = head_ops.get("HAN", 0)
        MAX_HAN = bw_len - hdn - tdn  # l_HA <= l_old-l_HD-l_TD
        valid_HAN = self.get_valid_distribution(self.prpm["HAN"], MAX_HAN)
        if han != 0:
            prob *= valid_HAN[han] / sum(valid_HAN.values())

        tan = tail_ops.get("TAN", 0)
        MAX_TAN = bw_len - hdn - tdn - han  # l_TA <= l_old-l_HD-l_TD-l_HA
        valid_TAN = self.get_valid_distribution(self.prpm["TAN"], MAX_TAN)
        if tan != 0:
            prob *= valid_TAN[tan] / sum(valid_TAN.values())

        HAC_sum = sum(self.prpm["HAC"].values())
        for i, c in enumerate(reversed(head_ops["HAC"])):
            if i < hdn:
                total = HAC_sum - self.prpm["HAC"][pw[hdn - i - 1]]
                prob *= self.prpm["HAC"][c] / total
            else:
                prob *= self.prpm["HAC"][c] / HAC_sum

        TAC_sum = sum(self.prpm["TAC"].values())
        for i, c in enumerate(tail_ops["TAC"]):
            if i < tdn:
                total = TAC_sum - self.prpm["TAC"][pw[tdn - i - 1]]
                prob *= self.prpm["TAC"][c] / total
            else:
                prob *= self.prpm["TAC"][c] / TAC_sum

        return prob

    def prpm_encode(self, bw, rules):
        codes = []
        len_bw = len(bw)

        head_ops = rules["H"]
        tail_ops = rules["T"]

        # Encode HDN
        hdn = head_ops.get("HDN", 0)
        MAX_HDN = len_bw // 2
        valid_HDN = self.get_valid_distribution(self.prpm["HDN"], MAX_HDN)
        valid_HDN_cdf = pdf2cdf(list(valid_HDN.values()))
        hdn_idx = list(valid_HDN.keys()).index(hdn)
        codes.append(self.dte.encode(hdn_idx, valid_HDN_cdf))

        # Encode TDN
        tdn = tail_ops.get("TDN", 0)
        MAX_TDN = len_bw // 2 - hdn
        valid_TDN = self.get_valid_distribution(self.prpm["TDN"], MAX_TDN)
        valid_TDN_cdf = pdf2cdf(list(valid_TDN.values()))
        tdn_idx = list(valid_TDN.keys()).index(tdn)
        codes.append(self.dte.encode(tdn_idx, valid_TDN_cdf))

        # Encode HAN
        han = head_ops.get("HAN", 0)
        MAX_HAN = len_bw - hdn - tdn
        valid_HAN = self.get_valid_distribution(self.prpm["HAN"], MAX_HAN)
        valid_HAN_cdf = pdf2cdf(list(valid_HAN.values()))
        han_idx = list(valid_HAN.keys()).index(han)
        codes.append(self.dte.encode(han_idx, valid_HAN_cdf))

        # Encode TAN
        tan = tail_ops.get("TAN", 0)
        MAX_TAN = len_bw - hdn - tdn - han
        valid_TAN = self.get_valid_distribution(self.prpm["TAN"], MAX_TAN)
        valid_TAN_cdf = pdf2cdf(list(valid_TAN.values()))
        tan_idx = list(valid_TAN.keys()).index(tan)
        codes.append(self.dte.encode(tan_idx, valid_TAN_cdf))

        # Encode HAC
        reversed_HAC = head_ops.get("HAC", [])[::-1]
        for j, c in enumerate(reversed_HAC):
            c_idx = list(self.charset).index(c)
            HAC_pdf = [self.prpm["HAC"].get(c, 0) for c in self.charset]

            if j == 0 and j < hdn:
                d_c_idx = list(self.charset).index(bw[hdn - j - 1])
                HAC_pdf_copy = HAC_pdf.copy()
                HAC_pdf_copy[d_c_idx] = 0
                HAC_cdf = pdf2cdf(HAC_pdf_copy)
                codes.append(self.dte.encode(c_idx, HAC_cdf))
            else:
                HAC_cdf = pdf2cdf(HAC_pdf)
                codes.append(self.dte.encode(c_idx, HAC_cdf))

        # Encode TAC
        for j, c in enumerate(tail_ops.get("TAC", [])):
            c_idx = list(self.charset).index(c)
            TAC_pdf = [self.prpm["TAC"].get(c, 0) for c in self.charset]

            if j == 0 and j < tdn:
                c_d_idx = list(self.charset).index(bw[len_bw - tdn + j])
                TAC_pdf_copy = TAC_pdf.copy()
                TAC_pdf_copy[c_d_idx] = 0
                TAC_cdf = pdf2cdf(TAC_pdf_copy)
                codes.append(self.dte.encode(c_idx, TAC_cdf))
            else:
                TAC_cdf = pdf2cdf(TAC_pdf)
                codes.append(self.dte.encode(c_idx, TAC_cdf))

        # Pad to fixed length
        if len(codes) < self.max_code_len:
            codes.extend(self.dte.padding(self.max_code_len - len(codes)))
        return codes

    def prpm_decode(self, bw, codes):
        len_bw = len(bw)
        rules = {
            "H": {"MOD": "None", "HDN": 0, "HAN": 0, "HAC": []},
            "T": {"MOD": "None", "TDN": 0, "TAN": 0, "TAC": []},
        }

        codes_copy = codes.copy()

        # Decode HDN
        MAX_HDN = len_bw // 2
        valid_HDN = self.get_valid_distribution(self.prpm["HDN"], MAX_HDN)
        valid_HDN_cdf = pdf2cdf(list(valid_HDN.values()))
        hdn_idx = self.dte.decode(codes_copy.pop(0), valid_HDN_cdf)
        hdn = list(valid_HDN.keys())[hdn_idx]
        rules["H"]["HDN"] = int(hdn)

        # Decode TDN
        MAX_TDN = len_bw // 2 - rules["H"]["HDN"]
        valid_TDN = self.get_valid_distribution(self.prpm["TDN"], MAX_TDN)
        valid_TDN_cdf = pdf2cdf(list(valid_TDN.values()))
        tdn_idx = self.dte.decode(codes_copy.pop(0), valid_TDN_cdf)
        tdn = list(valid_TDN.keys())[tdn_idx]
        rules["T"]["TDN"] = int(tdn)

        # Decode HAN
        MAX_HAN = len_bw - rules["H"]["HDN"] - rules["T"]["TDN"]
        valid_HAN = self.get_valid_distribution(self.prpm["HAN"], MAX_HAN)
        valid_HAN_cdf = pdf2cdf(list(valid_HAN.values()))
        han_idx = self.dte.decode(codes_copy.pop(0), valid_HAN_cdf)
        han = list(valid_HAN.keys())[han_idx]
        rules["H"]["HAN"] = int(han)

        # Decode TAN
        MAX_TAN = len_bw - rules["H"]["HDN"] - rules["T"]["TDN"] - rules["H"]["HAN"]
        valid_TAN = self.get_valid_distribution(self.prpm["TAN"], MAX_TAN)
        valid_TAN_cdf = pdf2cdf(list(valid_TAN.values()))
        tan_idx = self.dte.decode(codes_copy.pop(0), valid_TAN_cdf)
        tan = list(valid_TAN.keys())[tan_idx]
        rules["T"]["TAN"] = int(tan)

        # Decode HAC
        rules["H"]["HAC"] = []
        for j in range(rules["H"]["HAN"]):
            HAC_pdf = [self.prpm["HAC"].get(c, 0) for c in self.charset]
            if j == 0 and j < rules["H"]["HDN"]:
                d_c_idx = list(self.charset).index(bw[rules["H"]["HDN"] - j - 1])
                HAC_pdf_copy = HAC_pdf.copy()
                HAC_pdf_copy[d_c_idx] = 0
                HAC_cdf = pdf2cdf(HAC_pdf_copy)
                c_idx = self.dte.decode(codes_copy.pop(0), HAC_cdf)
                c = list(self.charset)[c_idx]
                rules["H"]["HAC"].insert(0, c)
            else:
                HAC_cdf = pdf2cdf(HAC_pdf)
                c_idx = self.dte.decode(codes_copy.pop(0), HAC_cdf)
                c = list(self.charset)[c_idx]
                rules["H"]["HAC"].insert(0, c)

        # Decode TAC
        rules["T"]["TAC"] = []
        for j in range(rules["T"]["TAN"]):
            TAC_pdf = [self.prpm["TAC"].get(c, 0) for c in self.charset]
            if j == 0 and j < rules["T"]["TDN"]:
                c_d_idx = list(self.charset).index(bw[len_bw - rules["T"]["TDN"] + j])
                TAC_pdf_copy = TAC_pdf.copy()
                TAC_pdf_copy[c_d_idx] = 0
                TAC_cdf = pdf2cdf(TAC_pdf_copy)
                c_idx = self.dte.decode(codes_copy.pop(0), TAC_cdf)
                c = list(self.charset)[c_idx]
                rules["T"]["TAC"].append(c)
            else:
                TAC_cdf = pdf2cdf(TAC_pdf)
                c_idx = self.dte.decode(codes_copy.pop(0), TAC_cdf)
                c = list(self.charset)[c_idx]
                rules["T"]["TAC"].append(c)

        # Set MOD based on the results of decoded H and T
        if rules["H"]["HDN"] == 0 and rules["H"]["HAN"] == 0:
            rules["H"]["MOD"] = "None"
        if rules["H"]["HDN"] > 0 and rules["H"]["HAN"] == 0:
            rules["H"]["MOD"] = "Delete"
        if rules["H"]["HDN"] == 0 and rules["H"]["HAN"] > 0:
            rules["H"]["MOD"] = "Add"
        if rules["H"]["HDN"] > 0 and rules["H"]["HAN"] > 0:
            rules["H"]["MOD"] = "Delete-Add"

        if rules["T"]["TDN"] == 0 and rules["T"]["TAN"] == 0:
            rules["T"]["MOD"] = "None"
        if rules["T"]["TDN"] > 0 and rules["T"]["TAN"] == 0:
            rules["T"]["MOD"] = "Delete"
        if rules["T"]["TDN"] == 0 and rules["T"]["TAN"] > 0:
            rules["T"]["MOD"] = "Add"
        if rules["T"]["TDN"] > 0 and rules["T"]["TAN"] > 0:
            rules["T"]["MOD"] = "Delete-Add"

        return rules

    def modify_head(self, pw, head_ops):
        if head_ops["MOD"] == "None":
            return pw
        elif head_ops["MOD"] == "Delete":
            return pw[head_ops["HDN"] :]
        elif head_ops["MOD"] == "Add":
            return "".join(head_ops["HAC"]) + pw
        elif head_ops["MOD"] == "Delete-Add":
            return "".join(head_ops["HAC"]) + pw[head_ops["HDN"] :]
        else:
            return pw

    def modify_tail(self, pw, tail_ops):
        if tail_ops["MOD"] == "None":
            return pw
        elif tail_ops["MOD"] == "Delete":
            return pw[: -tail_ops["TDN"]] if tail_ops["TDN"] > 0 else pw
        elif tail_ops["MOD"] == "Add":
            return pw + "".join(tail_ops["TAC"])
        elif tail_ops["MOD"] == "Delete-Add":
            return (pw[: -tail_ops["TDN"]] if tail_ops["TDN"] > 0 else pw) + "".join(
                tail_ops["TAC"]
            )
        else:
            return pw

    def get_valid_distribution(self, distribution, max_value):
        valid_distribution = {}
        for key, value in distribution.items():
            if int(key) <= max_value:
                valid_distribution[key] = value
        return valid_distribution

    def prpm_train(self, data, save_dir):
        prpm = {
            "H": {
                "None": 0,
                "Delete": 0,
                "Add": 0,
                "Delete-Add": 0,
            },
            "HDN": {l: self.pseudocount for l in range(self.l_max // 2 + 1)},
            "HAN": {l: self.pseudocount for l in range(self.l_max // 2 + 1)},
            "HAC": {c: self.pseudocount for c in self.charset},
            "T": {
                "None": 0,
                "Delete": 0,
                "Add": 0,
                "Delete-Add": 0,
            },
            "TDN": {l: self.pseudocount for l in range(self.l_max // 2 + 1)},
            "TAN": {l: self.pseudocount for l in range(self.l_max // 2 + 1)},
            "TAC": {c: self.pseudocount for c in self.charset},
        }
        for bw, pw, freq in tqdm(data, total=len(data), desc="Training PRPM"):
            rules = self.prpm_parse(bw, pw)
            if not rules:
                continue

            head_ops = rules["H"]
            tail_ops = rules["T"]

            hdn = head_ops["HDN"]
            tdn = tail_ops["TDN"]
            han = head_ops["HAN"]
            tan = tail_ops["TAN"]

            prpm["H"][head_ops["MOD"]] += freq
            prpm["T"][tail_ops["MOD"]] += freq

            prpm["HDN"][hdn] += freq
            prpm["TDN"][tdn] += freq
            prpm["HAN"][han] += freq
            prpm["TAN"][tan] += freq

            for c in head_ops["HAC"]:
                prpm["HAC"][c] += freq
            for c in tail_ops["TAC"]:
                prpm["TAC"][c] += freq

        print("Saving sppm model.")
        os.makedirs(save_dir, exist_ok=True)
        save_pickle(prpm, f"{save_dir}/prpm.pkl")

    def load_prpm_model(self, model_dir):
        self.prpm = load_pickle(f"{model_dir}/prpm.pkl")


class VaultGuard:
    def __init__(self, encoder):
        self.encoder = encoder

    def encode_vault(self, vault):
        reused = self.encoder.multi_reuse(vault)
        codes = []

        for i, pw in enumerate(vault):
            base_idx = self.encoder.reuse_check(i, reused)
            if base_idx is None:
                code = self.encoder.sppm_encode(pw)
                codes.append(code)
            else:
                bw = vault[base_idx]
                rules = self.encoder.prpm_parse(bw, pw)
                code = self.encoder.prpm_encode(bw, rules)
                codes.append(code)

        return [reused, codes]

    def decode_vault(self, reused, codes):
        vault = []

        for i, code in enumerate(codes):
            base_idx = self.encoder.reuse_check(i, reused)
            if base_idx is None:
                pw = self.encoder.sppm_decode(code)
                vault.append(pw)
            else:
                bw = vault[base_idx]
                rules = self.encoder.prpm_decode(bw, code)
                pw = self.encoder.prpm_generate(bw, rules)
                if not self.encoder.is_reuse(bw, pw):
                    raise ValueError(
                        "Generated password is not a reuse of base password."
                    )
                vault.append(pw)

        return vault

    def adaptive_encode_vault(self, vault, boosted_ngrams, alpha=2):
        reused = self.encoder.multi_reuse(vault)
        codes = []

        for i, pw in enumerate(vault):
            base_idx = self.encoder.reuse_check(i, reused)
            if base_idx is None:
                code = self.encoder.adaptive_sppm_encode(pw, boosted_ngrams, alpha)
                codes.append(code)
            else:
                bw = vault[base_idx]
                rules = self.encoder.prpm_parse(bw, pw)
                code = self.encoder.prpm_encode(bw, rules)
                codes.append(code)

        return [reused, codes]

    def adaptive_decode_vault(self, reused, codes, boosted_ngrams, alpha=2):
        vault = []

        for i, code in enumerate(codes):
            base_idx = self.encoder.reuse_check(i, reused)
            if base_idx is None:
                pw = self.encoder.adaptive_sppm_decode(code, boosted_ngrams, alpha)
                vault.append(pw)
            else:
                bw = vault[base_idx]
                rules = self.encoder.prpm_decode(bw, code)
                pw = self.encoder.prpm_generate(bw, rules)
                if self.encoder.is_reuse(bw, pw) is False:
                    raise ValueError(
                        "Generated password is not a reuse of base password."
                    )
                vault.append(pw)

        return vault

    def vault_prob(self, vault):
        reused = self.encoder.multi_reuse(vault)
        prob = 1.0

        for i, pw in enumerate(vault):
            base_idx = self.encoder.reuse_check(i, reused)
            if base_idx is None:
                pw_prob = self.encoder.sppm_prob(pw)
                prob *= pw_prob
            else:
                bw = vault[base_idx]
                pw_prob = self.encoder.prpm_prob(bw, pw)
                prob *= pw_prob

        return prob

    def adaptive_prob(self, bw, pw, boosted_ngrams, alpha=2):
        if bw is None:
            pw_prob = self.encoder.adaptive_sppm_prob(pw, boosted_ngrams, 5)
        else:
            pw_prob = self.encoder.prpm_prob(bw, pw, boosted_ngrams)
        return pw_prob

    def adaptive_vault_prob(self, reuse, vault, boosted_ngrams, alpha=2):
        log_prob = 0

        for i, pw in enumerate(vault):
            base_idx = self.encoder.reuse_check(i, reuse)
            if base_idx is None:
                pw_prob = self.encoder.adaptive_sppm_prob(pw, boosted_ngrams, alpha)
                log_prob += log(pw_prob)
            else:
                bw = vault[base_idx]
                pw_prob = self.encoder.prpm_prob(bw, pw, boosted_ngrams)
                log_prob += log(pw_prob)

        return log_prob


def init_vaultguard_nle(order=4, pseudocount=1.000):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dte = IS_DTE(128)
    encoder = Encoder(order, pseudocount, dte)
    encoder.load_sppm_model(f"{curr_dir}/model")
    encoder.load_prpm_model(f"{curr_dir}/model")
    nle = VaultGuard(encoder)
    return nle
