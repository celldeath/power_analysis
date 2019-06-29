"""
Power analysis for acute doxorubicin model
"""

import numpy as np
import pandas as pd
from scipy.stats import f_oneway

# Collecting data from pilot study
df = pd.read_csv('acute_male_fs.csv')
df.columns = ["saline", "dox"]
means = df.mean()
variances = df.var()
size = 5 # This number gives us 80% power at 5% significance level
alpha = 0.05 # Significance level
significant = []

# Parametric bootstrap
for _ in range(10000):
    # Draw new data from Normal distribution using prior parameters above
    draw_bs = np.random.multivariate_normal(means, np.diag(variances), size=size)
    df_bs  = pd.DataFrame(draw_bs, columns=means.index)

    # ANOVA test
    F, p = f_oneway(df_bs["saline"], df_bs["dox"])
    s = p < alpha
    significant.append(s)

# Check how often we were (correctly) able to reject the hypothesis.
print("Power: ", np.mean(significant))