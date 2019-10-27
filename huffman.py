class TreeNode:
    def __init__(self, name, prob, lchild=None, rchild=None):
        self.name = name
        self.prob = prob
        self.lchild = lchild
        self.rchild = rchild

    def __repr__(self):
        return f'Tree Name:{self.name} Prob:{self.prob}'


def encode(counts_dict):
    tree_nodes = [TreeNode(name, count, None, None)
                  for name, count in counts_dict.items()]
    return assign_codes(huffman_partition(tree_nodes))


def huffman_partition(tree_nodes):
    sorted_xs = sorted(tree_nodes, reverse=True, key=lambda x: x.prob)

    def helper(xs):
        rchild = xs.pop(-1)
        lchild = xs.pop(-1)
        # Insert into a sorted list.
        insort_wkey(xs, TreeNode(None,
                                 rchild.prob + lchild.prob, lchild, rchild),
                    key=lambda x: x.prob)
        if len(xs) == 1:
            return xs[0]
        return huffman_partition(xs)
    return helper(sorted_xs)


def assign_codes(tree):
    def assign_codes_helper(tree, code_lists, code=""):
        if tree.lchild is None and tree.rchild is None:
            code_dict[tree.name] = code
            return
        assign_codes_helper(tree.lchild, code_lists, code+'0')
        assign_codes_helper(tree.rchild, code_lists, code+'1')
    code_dict = {}
    assign_codes_helper(tree, code_dict)
    return code_dict


def reverse_dict(d):
    return dict(map(reversed, d.items()))


def decode(s, code_dict):
    # TCO FUNCTION:
    # def rec_decode(s, acc="", incoming=""):
    #     if len(s) == 0:
    #         return acc
    #     ind = invcodemap.get(incoming+s[0], None)
    #     if ind is not None:
    #         return rec_decode(s[1:], acc+possibilities[int(ind)], "")
    #     return rec_decode(s[1:], acc, incoming+s[0])
    # return rec_decode(s)
    # WHILE-LOOPED FUNCTION:
    invcodemap = reverse_dict(code_dict)
    acc = []
    incoming = ""
    while len(s) != 0:
        code = invcodemap.get(incoming+s[0], None)
        if code is not None:
            acc.append(code)
            s, incoming = s[1:], ""
        else:
            s, incoming = s[1:], incoming+s[0]
    return acc


def insort_wkey(a, x, key=None, lo=0, hi=None):
    # """Insert item x in list a, and keep it sorted assuming a is sorted.
    # If x is already in a, insert it to the right of the rightmost x.
    # Optional args lo (default 0) and hi (default len(a)) bound the
    # slice of a to be searched.
    # """
    # def bisect_wkey(a, x, key, lo=0, hi=None):
    """Return the index where to insert item x in list a, assuming a is sorted.
    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(x) will
    insert just after the rightmost x already there.
    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if key(x) < key(a[mid]):
            hi = mid
        else:
            lo = mid+1
    a.insert(lo, x)
