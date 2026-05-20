import colorsys
import numpy as np 

def sublist(lst1, lst2):
    return all([(x in lst2) for x in lst1])

def fmt_interval(interval):
    a, b = interval
    a_str = "-∞" if np.isneginf(a) else f"{a:.2f}"
    b_str = "+∞" if np.isposinf(b) else f"{b:.2f}"
    return f"[{a_str}, {b_str})"


def __rgb_to_hex(rgb):
    r, g, b = rgb
    return f"#{int(round(r * 255)):02x}{int(round(g * 255)):02x}{int(round(b * 255)):02x}"


def __generate_distinct_colors(n):
    if n <= 0:
        return []

    colors = []
    golden_ratio = 0.618033988749895
    h = 0.0
    for _ in range(n):
        h = (h + golden_ratio) % 1.0
        s = 0.65
        v = 0.9
        colors.append(colorsys.hsv_to_rgb(h, s, v))
    return colors


def __generate_distinct_hex_colors(n):
    return [__rgb_to_hex(rgb) for rgb in __generate_distinct_colors(n)]


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
                x_ctx.append(__get_interval_from_index(x_quantizer, idx, decimals))
            else:
                y_ctx.append(__get_interval_from_index(y_quantizer, idx, decimals))

        return f"Ctx X: {x_ctx}\nCtx Y: {y_ctx}"

    if quantizer is None:
        return f"Ctx: {node.context}"

    if len(node.context) == 0:
        return "Ctx: []"

    if hasattr(quantizer, 'decode_intervals'):
        decoded_ctx = []
        for idx in node.context:
            try:
                x_int, y_int = quantizer.decode_intervals(idx)
            except (TypeError, ValueError, IndexError, AttributeError):
                decoded_ctx.append(str(idx))
                continue
            decoded_ctx.append(f"(x:{fmt_interval(x_int)}, y:{fmt_interval(y_int)})")
        return "Ctx:\n" + "\n".join(decoded_ctx)

    if hasattr(quantizer, 'interval'):
        intervals = [__get_interval_from_index(quantizer, idx, decimals) for idx in node.context]
        return f"Ctx: {intervals}"

    if not hasattr(quantizer, 'unquantize'):
        return f"Ctx: {node.context}"

    try:
        decoded = [quantizer.unquantize(idx) for idx in node.context]
    except (TypeError, ValueError, IndexError, AttributeError):
        return f"Ctx: {node.context}"

    first = decoded[0]
    if hasattr(first, '__len__') and len(first) == 2:
        x_ctx = [round(float(v[0]), decimals) for v in decoded]
        y_ctx = [round(float(v[1]), decimals) for v in decoded]
        return f"Ctx X: {x_ctx}\nCtx Y: {y_ctx}"

    formatted = [round(float(v), decimals) for v in decoded]
    return f"Ctx: {formatted}"

def __get_interval_from_index(quantizer, index, decimals=2):
    """
    Convert a quantizer index to its corresponding interval [lower, upper).
    Returns a formatted interval string like '(-inf, 0.15)' or '[0.15, 0.30)'.
    """
    if not hasattr(quantizer, '_thresholds'):
        return str(index)
    
    thresholds = quantizer._thresholds
    n_thresholds = len(thresholds)
    
    # Handle bounds
    if index < 0 or index > n_thresholds:
        return str(index)
    
    lower = -float('inf') if index == 0 else thresholds[index - 1]
    upper = float('inf') if index == n_thresholds else thresholds[index]
    
    # Format with decimals
    lower_str = '-∞' if lower == -float('inf') else f'{lower:.{decimals}f}'
    upper_str = '+∞' if upper == float('inf') else f'{upper:.{decimals}f}'
    
    return f'[{lower_str}, {upper_str})'


def __collect_effective_regimes(node, min_observed_samples=1, regimes=None):
    if regimes is None:
        regimes = []

    if node.data.get('BS_len', 0) < min_observed_samples:
        return regimes

    if node.data.get('leaf_cbct', False):
        regimes.append(tuple(node.context))
        return regimes

    for child in node.children:
        __collect_effective_regimes(child, min_observed_samples=min_observed_samples, regimes=regimes)

    return regimes


def __normalize_context_map(context_map):
    if context_map is None:
        return None

    normalized_map = {}
    for context, regime_id in context_map.items():
        normalized_map[tuple(context)] = int(regime_id)

    return normalized_map


def __build_regime_map(root, min_observed_samples=1, context_map=None):
    normalized_context_map = __normalize_context_map(context_map)
    if normalized_context_map is not None:
        unique_regime_ids = sorted(set(normalized_context_map.values()))
        colors = __generate_distinct_hex_colors(len(unique_regime_ids))
        color_by_regime_id = {
            regime_id: colors[idx]
            for idx, regime_id in enumerate(unique_regime_ids)
        }
        return {
            context: {
                "id": regime_id,
                "fillcolor": color_by_regime_id[regime_id],
            }
            for context, regime_id in normalized_context_map.items()
        }

    regimes = __collect_effective_regimes(root, min_observed_samples=min_observed_samples)
    unique_regimes = []
    seen = set()
    for regime in regimes:
        if regime not in seen:
            unique_regimes.append(regime)
            seen.add(regime)

    colors = __generate_distinct_hex_colors(len(unique_regimes))
    return {
        regime: {
            "id": idx,
            "fillcolor": colors[idx],
        }
        for idx, regime in enumerate(unique_regimes)
    }


def __visualize_tree(node, dot=None, node_id=0, quantizer=None, decimals=2, quantizer_seq=None, x_quantizer=None, y_quantizer=None, min_observed_samples=1, regime_map=None):
    """
    Vizualise contextx and AR parameters (ms and variance) in each node of the tree.
    """
    if dot is None:
        dot = Digraph(comment='Tree')
        dot.attr(size="9,9", ratio="compress")
        dot.attr(
            "node",
            shape="circle",
            style="filled",
            fillcolor="#ffffff",
            color="#444444",
            fontsize="10",
            width="1.55",
            height="1.55",
            margin="0.06,0.04",
        )
        dot.attr("edge", color="#777777")

    # Hide branches under a configurable observation threshold.
    if node.data.get('BS_len', 0) < min_observed_samples:
        return dot, node_id

    ctx_str = __format_context_label(
        node,
        quantizer=quantizer,
        decimals=decimals,
        quantizer_seq=quantizer_seq,
        x_quantizer=x_quantizer,
        y_quantizer=y_quantizer,
    )


    label = f"{ctx_str}\n"
    if 'ms' in node.data:
        label += "AR coefficients : " + ", ".join([f"{m:.2f}" for m in list(node.data['ms'].flatten())]) + "\n"
    if 'var' in node.data:
        var = float(np.asarray(node.data['var']).squeeze()) # fix bug of array dimensions 
        label += f"σ : {var**0.5:.2f}\n"
    if 'BS_len' in node.data:
        label += f"Observed samples : {node.data['BS_len']}\n"

    node_attrs = {}
    regime = regime_map.get(tuple(node.context)) if regime_map is not None else None
    if regime is not None and node.data.get('leaf_cbct', False):
        label += f"Regime : R{regime['id']}\n"
        node_attrs.update({
            "fillcolor": regime["fillcolor"],
            "color": "#2f2f2f",
        })
    else:
        node_attrs.update({
            "fillcolor": "#ffffff",
            "color": "#777777",
        })
    current_id = str(node_id)
    dot.node(current_id, label, **node_attrs)
    
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
            regime_map,
        )
        if next_id != old_next_id:
            dot.edge(current_id, child_id)
        
    return dot, next_id
    


def view_tree_debug(tree: Tree):
    dot, _ = __visualize_tree_debug(tree.root)
    # dot.view()
    return dot

def view_tree(tree: Tree, quantizer=None, decimals=2, quantizer_seq=None, x_quantizer=None, y_quantizer=None, x_quantix=None, y_quantix=None, min_observed_samples=1, context_map=None):
    # Accept x_quantix/y_quantix typo as aliases for convenience.
    if x_quantizer is None and x_quantix is not None:
        x_quantizer = x_quantix
    if y_quantizer is None and y_quantix is not None:
        y_quantizer = y_quantix
    if isinstance(quantizer, dict):
        if x_quantizer is None:
            x_quantizer = quantizer.get("x_quantizer")
        if y_quantizer is None:
            y_quantizer = quantizer.get("y_quantizer")
        if quantizer_seq is None:
            quantizer_seq = quantizer.get("quantizer_seq")
        quantizer = quantizer.get("quantizer")

    regime_map = __build_regime_map(
        tree.root,
        min_observed_samples=min_observed_samples,
        context_map=context_map,
    )

    dot, _ = __visualize_tree(
        tree.root,
        quantizer=quantizer,
        decimals=decimals,
        quantizer_seq=quantizer_seq,
        x_quantizer=x_quantizer,
        y_quantizer=y_quantizer,
        min_observed_samples=min_observed_samples,
        regime_map=regime_map,
    )
    # dot.view()
    return dot
