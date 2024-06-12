import numpy as np
from scipy.stats import chi2_contingency, chisquare

# 卡方独立性检验
M = np.array([[762, 327, 468], [484, 239, 477]])
print("Matrix M:")
print("      Democrat Independent Republican")
print("F     ", "    ".join(map(str, M[0, :])))
print("M     ", "    ".join(map(str, M[1, :])))

result = chi2_contingency(M)
print("\nPearson's Chi-squared test")
print(f"X-squared = {result[0]:.2f}, df = {result[2]}, p-value = {result[1]:.3e}")

# 第二个例子
x = np.array([[12, 5], [7, 7]])
print("\nMatrix x:")
print(x)

result_x = chi2_contingency(x, correction=True)
print("\nPearson's Chi-squared test with Yates' continuity correction")
print(f"X-squared = {result_x[0]:.5f}, df = {result_x[2]}, p-value = {result_x[1]:.4f}")

result_x_no_correction = chi2_contingency(x, correction=False)
print("\nPearson's Chi-squared test")
print(f"X-squared = {result_x_no_correction[0]:.4f}, df = {result_x_no_correction[2]}, p-value = {result_x_no_correction[1]:.4f}")

# 卡方拟合优度检验
x = np.array([20, 15, 25])
result_fit = chisquare(x)
print("\nChi-squared test for given probabilities")
print(f"X-squared = {result_fit[0]:.1f}, df = {len(x)-1}, p-value = {result_fit[1]:.4f}")

x = np.array([89, 37, 30, 28, 2])
p = np.array([0.40, 0.20, 0.20, 0.19, 0.01])
result_fit_p = chisquare(x, f_exp=p*sum(x))
print("\nChi-squared test for given probabilities")
print(f"X-squared = {result_fit_p[0]:.4f}, df = {len(x)-1}, p-value = {result_fit_p[1]:.3f}")
