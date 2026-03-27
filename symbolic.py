import sympy as sp

# Define variables
y, c = sp.symbols('y c', real=True, positive=True)

# Define the function p(y)
p_y = (1 / (2 * y * sp.sqrt(2 * sp.pi * sp.ln(c / y)))) * sp.exp(-sp.ln(c / y) / 2)

# Compute the derivative
dp_dy = sp.diff(p_y, y)

# Display the derivative
sp.pprint(dp_dy)

# Evaluate at y = 0 (if possible)
limit_at_0 = sp.limit(dp_dy, y, 0)
print("Limit at y -> 0:", limit_at_0)
# Evaluate at y = 0.2
value_at_02 = dp_dy.subs(y, 0.2)
print("Value at y = 0.2:", value_at_02)