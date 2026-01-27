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
def __visualize_tree(node, dot=None, node_id=0):
    if dot is None:
        dot = Digraph(comment='Tree')

    data = dict(node.data)
    if 'BS' in data:
        del data['BS']
    label = str(data)
    current_id = str(node_id)
    dot.node(current_id, label)
    
    next_id = node_id + 1
    if node.is_leaf() is not False:
        return dot, next_id
    for child in node.children:
        child_id = str(next_id)
        dot.edge(current_id, child_id)
        dot, next_id = __visualize_tree(child, dot, next_id)
        
    return dot, next_id

def view_tree(tree: Tree):
    dot, _ = __visualize_tree(tree.root)
    dot.view()
    return dot
