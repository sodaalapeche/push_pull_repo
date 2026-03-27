import numpy as np
import math
import matplotlib.pyplot as plt
def linear_regression(log_y, log_p_y):
    # Convert inputs to numpy arrays for ease of calculation
    x = np.array(log_y)
    y = np.array(log_p_y)

    # Number of data points
    N = len(x)

    # Calculate the necessary sums
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x_squared = np.sum(x ** 2)

    # Calculate the slope (m) and intercept (b)
    m = (N * sum_xy - sum_x * sum_y) / (N * sum_x_squared - sum_x ** 2)
    b = (sum_y - m * sum_x) / N

    # Predicted values based on the linear model
    y_pred = m * x + b

    # Calculate residuals (errors)
    residuals = y - y_pred

    # Calculate R-squared
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum(residuals ** 2)
    r_squared = 1 - (ss_residual / ss_total)

    # Calculate standard error (SE)
    # Standard error of the regression (std_err) formula
    std_err = np.sqrt(ss_residual / (N - 2))

    # Calculate the t-statistic (slope / SE of slope)
    se_slope = std_err / np.sqrt(np.sum((x - np.mean(x)) ** 2))
    t_statistic = m / se_slope

    # Approximate p-value based on t-distribution for large N
    # For a rough estimate, we use the t-distribution formula:
    # p_value = 2 * (1 - CDF(t_statistic))
    # For large sample sizes, we can approximate p-value using a normal distribution:
    # p_value ≈ 2 * (1 - Normal CDF(t_statistic))
    # This approximation is generally acceptable for large N.
    p_value = 2 * (1 - normal_cdf(abs(t_statistic)))

    return m, b, r_squared, p_value, std_err


# Normal CDF approximation using a simple formula for large N (z-approximation)
def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / np.sqrt(2)))

def f(x, c=1):
    return c * np.exp(-x**2)

np.random.seed(42)
gaussian_data = np.random.normal(loc=0, scale=1, size=(100, 100))

pixel_values = gaussian_data.flatten()
mean = np.mean(pixel_values)
std_dev = np.std(pixel_values)
normalized_values = (pixel_values - mean) / std_dev

log_bins = np.logspace(np.log10(0.9), np.log10(10), num=100)
density, bins = np.histogram(np.abs(normalized_values), bins=log_bins, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2

y_vals = f(bin_centers)
valid_indices = (y_vals > 0) & (np.abs(bin_centers) > 0)
y_vals = y_vals[valid_indices]
density = density[valid_indices]

dx_dy = 1 / (2 * np.abs(bin_centers[valid_indices]) * y_vals)
p_y_exp = density * dx_dy

valid_prob_indices = np.isfinite(p_y_exp) & (p_y_exp > 0)
y_vals = y_vals[valid_prob_indices]
p_y_exp = p_y_exp[valid_prob_indices]

log_y = np.log(y_vals[:10])
log_p_y = np.log(p_y_exp[:10])

slope, intercept, r_value, p_value, std_err = linear_regression(log_y, log_p_y)
plt.figure()
plt.plot(log_y, log_p_y)
plt.xscale('log')
plt.yscale('log')
plt.show()
print(f"Pente initiale : {slope}")