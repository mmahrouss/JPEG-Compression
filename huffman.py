class TreeNode:
    '''
        Tree class used for the huffman coding
        Attributes:
            name: name of the symbol used only for leaves
            prob: probability or count of the symbol or tree of symbols
            lchild / rchild: pointer-like to the children of the tree
    '''

    def __init__(self, name, prob, lchild=None, rchild=None):
        self.name = name
        self.prob = prob
        self.lchild = lchild
        self.rchild = rchild


def encode(counts_dict):
    """
        Main encoding function for huffman
        takes the count or probability dictionary of mapping from
        symbols to count/prob.
        and returns a dictionary mapping symbols to binary code
        Args:
            counts_dict (Dict):  mapping from symbols to count/prob
        Returns:
            code_dict (Dict): mapping symbols to binary code
    """
    tree_nodes = [TreeNode(name, count, None, None)
                  for name, count in counts_dict.items()]
    return assign_codes(huffman_partition(tree_nodes))


def huffman_partition(tree_nodes):
    """
        Construct the huffman prefix tree
        Args:
            tree_nodes (list): list of singular trees representing the symbols
        Returns:
            htree (TreeNode): the huffman prefix tree 
    """
    sorted_xs = sorted(tree_nodes, reverse=True, key=lambda x: x.prob)

    def helper(xs):
        """ Recursive helper function for huffman """
        # Get the last two elements in the array
        # top two probabilities
        rchild = xs.pop(-1)
        lchild = xs.pop(-1)
        # Insert into a sorted list preserving the sorted state
        insort_wkey(xs, TreeNode(None,
                                 rchild.prob + lchild.prob, lchild, rchild),
                    key=lambda x: x.prob)
        if len(xs) == 1:
            # one tree left, we are done
            return xs[0]
        # continue partitioning on the remaining nodes
        return huffman_partition(xs)
    return helper(sorted_xs)


def assign_codes(tree):
    """
        Assign codes to symbols using huffman prefix tree
        Recursively traverses the tree (also called trie in this case) and 
        gets all the symbols and their codes and adds them to a hashed dict
        Args:
            tree: huffman prefix tree (trie)
        Returns:
            code_dict (Dict): Dictionary (also called hashed map)
                mapping symbols to binary codes
    """
    def assign_codes_helper(tree, code_lists, code=""):
        """ Helper recursive function """
        if tree.lchild is None and tree.rchild is None:
            # reached a leaf, add the mapped pair to the dictionary
            code_dict[tree.name] = code
            return
        # recursively apply to left subtree
        assign_codes_helper(tree.lchild, code_lists, code+'0')
        # and right subtree
        assign_codes_helper(tree.rchild, code_lists, code+'1')

    # initialize an empty dictionary
    code_dict = {}
    assign_codes_helper(tree, code_dict)
    return code_dict


def reverse_dict(d):
    """ reverses a dictionary """
    return dict(map(reversed, d.items()))


def decode(s, code_dict):
    """
        Decodes a huffman encoded sequence.
        Uses the generated dictionary that maps from symbol to code.
        Args:
            s (str): string of '0's and '1's
            code_dict (dict): maps from symbol to code
    """
    # reverse the dictionary
    invcodemap = reverse_dict(code_dict)
    result = []
    incoming = ""  # accumulates bits until we get a dictionary hit
    while len(s) != 0:
        code = invcodemap.get(incoming+s[0], None)
        if code is not None:
            # found code in dictionary, add it to result
            result.append(code)
            s, incoming = s[1:], ""
        else:
            # not found, get another bit
            s, incoming = s[1:], incoming+s[0]
    return result


def insort_wkey(a, x, key=lambda x: x, lo=0, hi=None):
    """
    Binary insert using a key (function) to be applied before comparison
    and keep it sorted assuming a is sorted.
    If x is already in a, insert it to the right of the rightmost x.
    Return the index where to insert item x in list a, assuming a is sorted.
    The return value i is such that all key(e) in a[:i] have key(e) <= x, and
    all key(e) in a[i:] have key(e) > x.
    Args:
        a (list): list to insert into
        x : element to insert
    Optional args
        lo (default 0)
        hi (default len(a))
        bound the slice of a to be searched.
    returns:
        None
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
