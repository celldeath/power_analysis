"""
Power analysis for breast cancer xenograft model
"""

import numpy as np
import pandas as pd
from scipy.stats import f_oneway

df = pd.read_csv('mouse_tumor_volume.csv')
means = df.mean()
variances = np.square(1.5 * df.std())  # 1.5x standard deviation (conservative)
size = 9  # This number gives us >80% power at 5% significance level
alpha = 0.05  # Significance level
significant = []

# Parametric bootstrap
for _ in range(10000):
    # Draw new data from Normal distribution using prior parameters above
    draw_bs = np.random.multivariate_normal(means, np.diag(variances), size=size)
    df_bs = pd.DataFrame(draw_bs, columns=means.index)

    # ANOVA test
    F, p = f_oneway(df_bs["saline"], df_bs["dox"], df_bs["dox+bai1"])
    s = p < alpha
    significant.append(s)

# Check how often we were (correctly) able to reject the hypothesis.
print("Power: ", np.mean(significant))  # ~91%
