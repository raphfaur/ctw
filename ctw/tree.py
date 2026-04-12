def sublist(lst1, lst2):
    return all([(x in lst2) for x in lst1])

class TreeNode :
    def __init__(self, children : list = None):
        self.children = children if children is not None else []
        self._value = None
        self._context = None
        self.data = {}
    
    def is_leaf(self) -> bool :
        return len(self.children) == 0

    @property
    def context(self) :
        return self._context
    
    @context.setter
    def context(self, ctx : list) :
        self.data['context'] = ctx
        self._context = ctx

    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, val) :
        self.data['value'] = val
        self._value = val

    @staticmethod
    def build_from_string(s : list, depth : int, alphabet : list, context : list = None) :
        current_node = TreeNode()
        current_context = context[::] if context is not None else []
        current_node.context = current_context

        if len(current_context) :
            current_node.value = current_context[-1]

        has_children = False
        for char in alphabet :
            current_context.append(char)
            a = ''.join(map(str, current_context))[::-1]
            b = ''.join(map(str, s))
            if a in b :
                has_children = True
            current_context.pop()

        if not has_children or depth == 0 : 
            return current_node

        children = []
        for char in alphabet :
            current_context.append(char)
            node = TreeNode.build_from_string(s, depth - 1, alphabet, current_context)
            current_context.pop()
            children.append(node)
            current_node.children = children
        
        return current_node

    @staticmethod
    def build_max_tree(depth : int, alphabet : list, context : list = None) :
        current_node = TreeNode()
        current_context = context[::] if context is not None else []
        current_node.context = current_context

        if len(current_context) :
            current_node.value = current_context[-1]

        if depth == 0 :
            return current_node

        children = []
        for char in alphabet :
            current_context.append(char)
            node = TreeNode.build_max_tree(depth - 1, alphabet, current_context)
            current_context.pop()
            children.append(node)
            current_node.children = children
        
        return current_node

    @staticmethod
    def build_bivariate_tree(depth : int, x_alphabet : list, y_alphabet : list, quantizer_seq : list = None, context : list = None) :
        current_node = TreeNode()
        current_context = context[::] if context is not None else []
        current_node.context = current_context

        if len(current_context) :
            current_node.value = current_context[-1]

        if depth == 0 :
            return current_node

        children = []
        current_quantizer = quantizer_seq[-depth]
        current_alphabet = x_alphabet if current_quantizer == 0 else y_alphabet
        for char in current_alphabet :
            current_context.append(char)
            node = TreeNode.build_bivariate_tree(depth - 1, x_alphabet, y_alphabet, quantizer_seq, current_context)
            current_context.pop()
            children.append(node)
            current_node.children = children
        
        return current_node

class Tree :
    def __init__(self, m : int, root : 'TreeNode') -> None :
        self.m = m
        self.root = root if root is not None else TreeNode([])
    
    def build_from_string(s : list, depth : int, alphabet : list) -> 'Tree' :
        root = TreeNode.build_from_string(s, depth, alphabet)
        return Tree(m=depth, root=root)

    def build_max_tree(depth : int, alphabet : list) -> 'Tree' :
        root = TreeNode.build_max_tree(depth, alphabet, [])
        return Tree(m=depth, root=root)
    
    def build_bivariate_tree(x_alphabet : list, y_alphabet : list, quantizer_seq : list = None) -> 'Tree' :
        depth = len(quantizer_seq) if quantizer_seq is not None else 0
        root = TreeNode.build_bivariate_tree(depth, x_alphabet, y_alphabet, quantizer_seq, [])
        return Tree(m=depth, root=root)





## Vizualization utils

from anytree import Node, RenderTree
def __to_anytree(node, name="Root"):
    display_name = node.value if node.value else name
    any_node = Node(display_name)
    for child in node.children:
        child_any = __to_anytree(child)
        child_any.parent = any_node
    return any_node

def print_tree(tree: Tree):
    root_anytree = __to_anytree(tree.root)
    for pre, fill, node in RenderTree(root_anytree):
        print(f"{pre}{node.name}")


from graphviz import Digraph
def __visualize_tree_debug(node, dot=None, node_id=0):
    if dot is None:
        dot = Digraph(comment='Tree')

    data = dict(node.data)
    if 'BS' in data:
        del data['BS']
    label = str(data)
    current_id = str(node_id)
    dot.node(current_id, label)
    
    next_id = node_id + 1
    if node.data['leaf_cbct'] is True:
        return dot, next_id
    for child in node.children:
        child_id = str(next_id)
        dot.edge(current_id, child_id)
        dot, next_id = __visualize_tree_debug(child, dot, next_id)
        
    return dot, next_id


def __format_context_label(node, quantizer=None, decimals=2, quantizer_seq=None, x_quantizer=None, y_quantizer=None):
    # Bivariate decoding mode where each level uses either X or Y quantizer.
    if quantizer_seq is not None and x_quantizer is not None and y_quantizer is not None:
        if len(node.context) == 0:
            return "Ctx X: []\nCtx Y: []"
        if len(node.context) > len(quantizer_seq):
            return f"Ctx: {node.context}"

        x_ctx = []
        y_ctx = []
        for i, idx in enumerate(node.context):
            q_selector = quantizer_seq[i]
            if q_selector == 0:
                x_ctx.append(round(float(x_quantizer.unquantize(idx)), decimals))
            else:
                y_ctx.append(round(float(y_quantizer.unquantize(idx)), decimals))

        return f"Ctx X: {x_ctx}\nCtx Y: {y_ctx}"

    if quantizer is None:
        return f"Ctx: {node.context}"

    if not hasattr(quantizer, 'unquantize'):
        return f"Ctx: {node.context}"

    try:
        decoded = [quantizer.unquantize(idx) for idx in node.context]
    except (TypeError, ValueError, IndexError, AttributeError):
        return f"Ctx: {node.context}"

    if len(decoded) == 0:
        return "Ctx X: []\nCtx Y: []"

    first = decoded[0]
    if hasattr(first, '__len__') and len(first) == 2:
        x_ctx = [round(float(v[0]), decimals) for v in decoded]
        y_ctx = [round(float(v[1]), decimals) for v in decoded]
        return f"Ctx X: {x_ctx}\nCtx Y: {y_ctx}"

    formatted = [round(float(v), decimals) for v in decoded]
    return f"Ctx: {formatted}"


def __visualize_tree(node, dot=None, node_id=0, quantizer=None, decimals=2, quantizer_seq=None, x_quantizer=None, y_quantizer=None, min_observed_samples=1):
    """
    Vizualise contextx and AR parameters (ms and variance) in each node of the tree.
    """
    if dot is None:
        dot = Digraph(comment='Tree')

    # Hide branches under a configurable observation threshold.
    if node.data.get('BS_len', 0) < min_observed_samples:
        return dot, node_id

    label = __format_context_label(
        node,
        quantizer,
        decimals=decimals,
        quantizer_seq=quantizer_seq,
        x_quantizer=x_quantizer,
        y_quantizer=y_quantizer,
    ) + "\n"
    if 'ms' in node.data:
        label += "AR coefficients : " + ", ".join([f"{m:.2f}" for m in list(node.data['ms'].flatten())]) + "\n"
    if 'var' in node.data:
        label += f"σ : {float(node.data['var']**0.5):.2f}\n"
    if 'BS_len' in node.data:
        label += f"Observed samples : {node.data['BS_len']}\n"
    current_id = str(node_id)
    dot.node(current_id, label)
    
    next_id = node_id + 1
    if node.data['leaf_cbct'] is True:
        return dot, next_id
    
    for child in node.children:
        child_id = str(next_id)
        old_next_id = next_id
        dot, next_id = __visualize_tree(
            child,
            dot,
            next_id,
            quantizer,
            decimals,
            quantizer_seq,
            x_quantizer,
            y_quantizer,
            min_observed_samples,
        )
        if next_id != old_next_id:
            dot.edge(current_id, child_id)
        
    return dot, next_id
    

def view_tree_debug(tree: Tree):
    dot, _ = __visualize_tree_debug(tree.root)
    # dot.view()
    return dot

def view_tree(tree: Tree, quantizer=None, decimals=2, quantizer_seq=None, x_quantizer=None, y_quantizer=None, x_quantix=None, y_quantix=None, min_observed_samples=1):
    # Accept x_quantix/y_quantix typo as aliases for convenience.
    if x_quantizer is None and x_quantix is not None:
        x_quantizer = x_quantix
    if y_quantizer is None and y_quantix is not None:
        y_quantizer = y_quantix

    dot, _ = __visualize_tree(
        tree.root,
        quantizer=quantizer,
        decimals=decimals,
        quantizer_seq=quantizer_seq,
        x_quantizer=x_quantizer,
        y_quantizer=y_quantizer,
        min_observed_samples=min_observed_samples,
    )
    # dot.view()
    return dot




