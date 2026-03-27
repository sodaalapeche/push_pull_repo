import numpy as np
import matplotlib.pyplot as plt
import math
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


def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / np.sqrt(2)))

def f(x):
    return 1/ (2*np.sqrt(2*x*math.pi*np.log(1/x)))

x_values = np.linspace(0.001, 1, 1000)  # Logarithmically spaced x values from 1e-10 to 1
x_values = x_values[x_values < 1]  # Exclude x >= 1 to avoid invalid log(1/x)

y_values = f(x_values)
xlog=np.log(x_values)
y_log=np.log(y_values)
slope, intercept, r_value, p_value, std_err = linear_regression(xlog[0:10], y_log[0:10])


yfit=x_values[10:1000]*slope+np.exp(intercept)
# Create the log-log plot
plt.plot(x_values, y_values, label=r"$p(y) = \frac{1}{2y\sqrt{2\pi \ln \left(\frac{c}{y}\right)}} \exp\left(-\frac{\ln \left(\frac{c}{y}\right)}{2}\right)$")
plt.plot(xlog[10:1000], yfit, label=r"$y->0⁺, tangent \approx $"+f"${str(slope)[0:5]}$")
plt.xlabel(r"$\log(y)$")
plt.xscale("log")
plt.yscale("log")
plt.ylabel(r"$\log(p(y))$")

plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.show()
print(slope, intercept, r_value, p_value, std_err)