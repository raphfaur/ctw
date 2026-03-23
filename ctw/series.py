
import numpy as np

class AR :  
    """
    AutoRegressive model.
    Example of usage :
    model = AR(p=2, coeffs=[0.5, -0.3], sigma=1, n_sample=100)
    x = [value for value in model] ## Creates a 100 samples sequence.
    Attributes:
        p (int): order of the model
    """
    
    def __init__(self, p : int, coeffs : list, sigma : float = 1, n_sample : int = 100, constant_term = 0) -> None :
        self.p = p
        self._history = [0] * p
        self.sigma = sigma
        self._n_sample = n_sample
        self.sample_k = 0
        self.constant_term = constant_term

        if len(coeffs) != p :
            raise ValueError("Number of coefficients must be equal to the order p")
        self.coeffs = coeffs

    def __next__(self) -> float :
        if self.sample_k >= self._n_sample :
            raise StopIteration
        next_value = sum(self.coeffs[i] * self._history[-(i+1)] for i in range(self.p))
        if self.constant_term :
            next_value += self.constant_term
        next_value += np.random.normal(0, self.sigma)
        self._history.append(next_value)
        self._history.pop(0)
        self.sample_k += 1
        return next_value
    
    def __iter__(self):
        return self