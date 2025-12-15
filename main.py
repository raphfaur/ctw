#%%
import ctw.series as series

# %%
ar_model = series.AR(p=2, coeffs=[0.7, -0.3], sigma=0.5, n_sample=10)
serie = [value for value in ar_model]
print(serie)


# %%
import matplotlib.pyplot as plt
plt.plot([value for value in series.AR(p=2, coeffs=[0.7, -0.3], sigma=0.5, n_sample=1000)])
plt.title("AutoRegressive (AR) Model Time Series")
# %%

# %%
import ctw.models as models
quantizer = models.Quantizer(thresholds=[-1, 0, 1, 2, 3, 4, 5])
assert(quantizer.quantize(2.5) == 4)
assert(quantizer.quantize(-2) == 0) 
assert(quantizer.quantize(-0.9) == 1) 
# %%
