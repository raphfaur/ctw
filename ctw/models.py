from .tree import Tree, TreeNode
import numpy as np
from scipy.special import gammaln

class Quantizer :
    """
    Quantizes a continuous value into discrete values based on predefined thresholds.
    """
    def __init__(self, thresholds : list = range(10)) -> None :
        self._thresholds = thresholds

    def quantize(self, value: float) -> int :
        for i, threshold in enumerate(self._thresholds) :
            if value < threshold :
                break
        return i

    def get_alphabet(self):
        return range(len(self._thresholds))
    
    def unquantize(self, index : int) -> float :
        return self._thresholds[index]


class LiveARTree() :
    """
    The LiveARTree class implements an online Context Tree Weighting model for AutoRegressive processes.
    The main difference with ARTree is that this model can be updated incrementally with new observations.
    This versio is prefered and should give the same results as ARTree when fed the same sequence of observations.
    """
    def __init__(self, max_depth : int, quantizer : Quantizer = Quantizer(), order = 1, init_sequence = None, tau = 1.0, lambda_ = 1.0, beta = 0.8) -> None :
        ## Prior hyperparameters for the AR model

        ## Order of the AR model 
        self.order = order

        ## Inv gamma for the noise variance
        self._tau = tau
        self._lambda = lambda_

        ## Gaussian for the AR parameters (p parameter)
        self._mu_0 = np.zeros((order, 1))
        self._Sigma_0 = np.diag(np.ones(order))

        ## Hyperparameters of the tree
        ## Besta must be > 1/2 according to the paper
        self.beta = beta
        self.max_depth = max_depth

        ## Tree
        self.alphabet = quantizer.get_alphabet()
        self.tree = Tree.build_max_tree(self.max_depth, self.alphabet)
        self._init_tree(self.tree.root)

        self.quantizer = quantizer

        ## Save observations
        self._x = init_sequence if init_sequence is not None else []

    def _init_tree(self, node : TreeNode) -> None :
        node.data.update({
            ## CWT
            'log_P_w,s' : 0.0,
            ## CCBT
            'log_P_m,s' : 0.0,
            ## P_e data
            'BS_len' : 0,
            's1' : 0.0,
            's2' : np.zeros((self.order, 1)),
            'S3' : np.zeros((self.order, self.order)),
            'log_Pe' : 0.0,
            ## AR parameters
            'ms' : np.zeros((self.order, 1)),
            'var' : 0.0,
            ## Utils
            'sum_log_P_w,s_children' : 0.0,
            'sum_log_P_m,s_children' : 0.0,
            'leaf_ctw' : True,
            'leaf_cbct' : True,
            'D_s' : 0.0, ## Used to compute var
        })
        for child in node.children :
            self._init_tree(child)

    def _update_tree_quantities(self, node : TreeNode, x_new : float, history : list, context_to_navigate : list, depth : int) -> None :
        ## Update current node quantities
        node.data['BS_len'] += 1
        x_i_tilde = np.array(history).reshape(1, self.order)
        node.data['s1'] += x_new ** 2
        node.data['s2'] += x_new * x_i_tilde.T
        node.data['S3'] += np.outer(x_i_tilde, x_i_tilde)
        if depth < self.max_depth :
            node.data['leaf_ctw'] = False
            node.data['leaf_cbct'] = False

        self._log_S_e(node)

        for child in node.children :
            if child.value == context_to_navigate[depth] :
                self._update_tree_quantities(child, x_new, history, context_to_navigate, depth + 1)

    def _update_tree_cctw(self, node : TreeNode, x_new : float, context_to_navigate : list, depth : int) -> None :
        if node.data['leaf_ctw'] :
            node.data['log_P_w,s'] = node.data['log_Pe']
        else :
            context_index = depth
            context_value = context_to_navigate[context_index]

            old_child_Pws = 0.0
            new_child_Pws = 0.0
            for child in node.children :
                if child.value == context_value :
                    old_child_Pws = child.data['log_P_w,s']
                    self._update_tree_cctw(child, x_new, context_to_navigate, depth + 1)
                    new_child_Pws = child.data['log_P_w,s']
                    break
            
            sum_log_children = node.data['sum_log_P_w,s_children'] - old_child_Pws
            sum_log_children += new_child_Pws
            log_pws = np.logaddexp(np.log(self.beta) + node.data['log_Pe'],
                                   np.log(1 - self.beta) + sum_log_children)
            node.data['log_P_w,s'] = log_pws
            node.data['sum_log_P_w,s_children'] = sum_log_children
        
        self._compute_ms(node)
        self._compute_var(node)

    def _update_tree_cbct(self, node : TreeNode, x_new : float, context_to_navigate : list, depth : int) -> None :
        if node.data['leaf_ctw'] :
            # print("leaf ctw, depth :", depth, node.context)
            if depth == self.max_depth :
                # print(depth, node.context)
                node.data['log_P_m,s'] = node.data['log_Pe']
            else:
                node.data['log_P_m,s'] = np.log(self.beta)
        else :
            context_index = depth
            context_value = context_to_navigate[context_index]

            old_child_Pms = 0.0
            for child in node.children :
                if child.value == context_value :
                    old_child_Pms = child.data['log_P_m,s']
                    self._update_tree_cbct(child, x_new, context_to_navigate, depth + 1)
                    break
            
            sum_log_children = node.data['sum_log_P_m,s_children'] - old_child_Pms
            sum_log_children += child.data['log_P_m,s']
            node.data['sum_log_P_m,s_children'] = sum_log_children

            a = np.log(self.beta) + node.data['log_Pe']
            b = np.log(1 - self.beta) + sum_log_children

            if a > b :
                node.data['leaf_cbct'] = True
                node.data['log_P_m,s'] = a
            else :
                node.data['leaf_cbct'] = False
                node.data['log_P_m,s'] = b

    def _update_tree(self, node : TreeNode, x_new : float, history : list, context_to_navigate : list , depth : int) -> None :
        self._update_tree_quantities(node, x_new, history, context_to_navigate, depth)
        self._update_tree_cctw(node, x_new, context_to_navigate, depth)
        self._update_tree_cbct(node, x_new, context_to_navigate, depth)

    def _log_S_e(self, leaf : TreeNode) -> float :
        n = leaf.data['BS_len']

        # S1 is a scalar
        s1 = leaf.data['s1']

        # s2 is a vector
        s2 = leaf.data['s2']
        
        # S3 is a matrix
        S3 = leaf.data['S3']

        a = (np.linalg.inv(self._Sigma_0) @ self._mu_0)
        b = np.linalg.inv(S3 + np.linalg.inv(self._Sigma_0))
        D_s = s1 + self._mu_0.T @ np.linalg.inv(self._Sigma_0) @ self._mu_0 - a.T @ b @ a

        Lambda_0 = np.linalg.inv(self._Sigma_0)
        Lambda_n = S3 + Lambda_0
        
        # Get log det
        _, log_det_Sigma0 = np.linalg.slogdet(self._Sigma_0)
        _, log_det_LambdaN = np.linalg.slogdet(Lambda_n)

        # log(C_s)
        log_Cs = 1/2 * ( (n * np.log(2 * np.pi)) + log_det_Sigma0 + log_det_LambdaN)

        # log of the ratio
        gamma_term = gammaln(self._tau + n/2) - gammaln(self._tau)
        lambda_term = (self._tau * np.log(self._lambda)) - ((self._tau + n/2) * np.log(self._lambda + D_s/2))

        # 1/Cs * gamma_term * lambda_term
        log_Pe = -log_Cs + gamma_term + lambda_term

        # save data in node
        leaf.data['log_Pe'] = log_Pe
        leaf.data['D_s'] = D_s
        
        return log_Pe

    def _compute_ms(self, node : TreeNode) -> None :
        s3 = node.data['S3']
        s2 = node.data['s2']
        ms = np.linalg.inv(s3 + np.linalg.inv(self._Sigma_0)) @ (s2 + np.linalg.inv(self._Sigma_0) @ self._mu_0)
        node.data['ms'] = ms
    
    def _compute_var(self, node : TreeNode) -> None :
        D_s = node.data['D_s']
        BS_len = node.data['BS_len']
        var = (2 * self._lambda + D_s) / (2 * self._tau + BS_len + 2)
        node.data['var'] = var

    def observe(self, x_new : float) -> None :
        self._x.append(x_new)
        if len(self._x) < self.max_depth + 1:
            return

        if self.tree is None :
            self.tree = Tree.build_max_tree(self.max_depth, self.alphabet)

        context_to_navigate = [self.quantizer.quantize(v) for v in self._x[-self.max_depth-1:-1]][::-1]
        # print(context_to_navigate)
        history = self._x[-self.order-1:-1][::-1]
        # print(history)
        self._update_tree(self.tree.root, x_new, history, context_to_navigate, 0)
    
    def predict(self, history : list) -> dict :
        if len(history) < self.order :
            raise ValueError("History length must be at least equal to the order of the AR model.")
        history = history[::-1]
        context_to_navigate = [self.quantizer.quantize(v) for v in history[:self.max_depth]][::-1]
        node = self.tree.root
        depth = 0
        while not node.data['leaf_cbct'] and depth < self.max_depth :
            context_value = context_to_navigate[depth]
            found_child = False
            for child in node.children :
                if child.value == context_value :
                    node = child
                    found_child = True
                    break
            if not found_child :
                break
            depth += 1
        ## Use the ar parameters stored in the node to make prediction
        mean = node.data['ms']
        x_new = float(mean.T @ np.array(history).reshape(self.order, 1))

        return x_new
    
    def predict_cctw(self, history: list) -> float:
        """
        CCTW : renvoie l'espérance conditionnelle et non la prédiction MAP
        """
        if len(history) < self.order:
            raise ValueError("History length must be at least equal to the order of the AR model.")

        # Même prétraitement que predict()
        history = history[::-1]
        context_to_navigate = [
            self.quantizer.quantize(v)
            for v in history[:self.max_depth]
        ][::-1]

        # Parcours du chemin suffixe unique
        node = self.tree.root
        depth = 0
        path_nodes = [node]

        while depth < self.max_depth:
            context_value = context_to_navigate[depth]
            found_child = False

            for child in node.children:
                if child.value == context_value:
                    node = child
                    path_nodes.append(node)
                    found_child = True
                    break

            if not found_child:
                break

            depth += 1

        # Poids CCTW
        log_weights = np.array(
            [float(n.data['log_P_w,s']) for n in path_nodes],
            dtype=float
        )

        # stabilité numérique
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights /= np.sum(weights)

        # Espérance conditionnelle = moyenne pondérée des AR locaux
        h = np.array(history[:self.order], dtype=float).reshape(self.order, 1)

        x_pred = 0.0
        for w, n in zip(weights, path_nodes):
            ms = np.array(n.data['ms'], dtype=float)
            mu = float(ms.T @ h)
            x_pred += float(w) * mu

        return float(x_pred)

