
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
        

class ARTree :
    def __init__(self, max_depth : int, quantizer : Quantizer = Quantizer()) -> None :
        self.max_depth = max_depth
        self._order = len(quantizer._thresholds)
        self.tree