
import numpy as np

class AR :  
    """
    AutoRegressive model

    Attributes:
        p (int): order of the model
    """
    
    def __init__(self, p : int, coeffs : list, sigma : float = 1, n_sample : int = 100) -> None :
        self.p = p
        self._history = [0] * p
        self.sigma = sigma
        self._n_sample = n_sample
        self.sample_k = 0

        if len(coeffs) != p :
            raise ValueError("Number of coefficients must be equal to the order p")
        self.coeffs = coeffs

    def __next__(self) -> float :
        if self.sample_k >= self._n_sample :
            raise StopIteration
        next_value = sum(self.coeffs[i] * self._history[-(i+1)] for i in range(self.p))
        next_value += np.random.normal(0, self.sigma)
        self._history.append(next_value)
        self._history.pop(0)
        self.sample_k += 1
        return next_value
    
    def __iter__(self):
        return self