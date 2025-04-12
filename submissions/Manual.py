import numpy as np
from scipy.optimize import minimize_scalar

profits = np.array([93750, 96429, 133500, 130500, 127500, 124500, 123000, 120000, 118500, 115500, 136875, 131250, 112500, 103125, 97500, 117500, 102500, 87500, 75000, 67500, 90000, 78750, 63750, 56250, 45000])
actual = np.array([4.9, 3.4, 10.8, 9.8, 6.5, 5.4, 6.2, 4.1, 5.4, 4.6, 11.3, 8.2, 3.7, 2.6, 1.9, 3.0, 1.9, 1.2, 0.6, 0.6, 1.5, 0.6, 0.6, 0.6, 0.6])

def error_function(k):
    transformed = profits ** k
    predicted = (transformed / transformed.sum()) * 100
    # Weight errors by actual percentages (prioritize high values)
    weights = actual / actual.sum()  # or use weights = actual directly
    sse = np.sum(weights * (predicted - actual) ** 2)
    return sse

result = minimize_scalar(error_function, bounds=(1, 10), method='bounded')
optimal_k = result.x
print(f"Optimal k (weighted SSE) = {optimal_k:.4f}")

def error_function_softmax(T):
    scaled = profits / T  # scale by temperature
    exp_values = np.exp(scaled - np.max(scaled))  # avoid overflow
    predicted = (exp_values / exp_values.sum()) * 100
    sse = np.sum((predicted - actual) ** 2)
    return sse

result = minimize_scalar(error_function_softmax, bounds=(1000, 50000), method='bounded')
optimal_T = result.x
print(f"Optimal T = {optimal_T:.0f}")