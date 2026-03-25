import numpy as np 

def sublist(lst1, lst2):
    return all([(x in lst2) for x in lst1])

def fmt_interval(interval):
    a, b = interval
    a_str = "-∞" if np.isneginf(a) else f"{a:.2f}"
    b_str = "+∞" if np.isposinf(b) else f"{b:.2f}"
    return f"[{a_str}, {b_str})"


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


def __visualize_tree(node, quantizer=None, dot=None, node_id=0):
    """
    Vizualise contextx and AR parameters (ms and variance) in each node of the tree.
    """
    if dot is None:
        dot = Digraph(comment='Tree')

    # added context decoding 
    if quantizer is not None and len(node.context) > 0:
        decoded_ctx = []
        for idx in node.context:
            x_int, y_int = quantizer.decode_intervals(idx)
            decoded_ctx.append(f"(x:{fmt_interval(x_int)}, y:{fmt_interval(y_int)})")
        ctx_str = "\n".join(decoded_ctx)
    else:
        ctx_str = str(node.context)



    label = f"Ctx:\n{ctx_str}\n"

    if 'ms' in node.data:
        label += "AR coefficients : " + ", ".join([f"{m:.2f}" for m in list(node.data['ms'].flatten())]) + "\n"
    if 'var' in node.data:
        var = float(np.asarray(node.data['var']).squeeze()) # fix bug of array dimensions 
        label += f"σ : {var**0.5:.2f}\n"
    if 'BS_len' in node.data:
        label += f"Observed samples : {node.data['BS_len']}\n"
    current_id = str(node_id)
    dot.node(current_id, label)
    
    next_id = node_id + 1
    if node.data['leaf_cbct'] is True:
        return dot, next_id
    
    for child in node.children:
        child_id = str(next_id)
        dot.edge(current_id, child_id)
        dot, next_id = __visualize_tree(child, quantizer, dot, next_id)
        
    return dot, next_id
    


def view_tree_debug(tree: Tree):
    dot, _ = __visualize_tree_debug(tree.root)
    # dot.view()
    return dot

def view_tree(tree: Tree, quantizer=None):
    dot, _ = __visualize_tree(tree.root,quantizer=quantizer)
    # dot.view()
    return dot




