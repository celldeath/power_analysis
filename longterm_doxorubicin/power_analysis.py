"""
Power analysis for long-term doxorubicin model
"""

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols

# Collecting data from wild type
df = pd.read_csv('chronic_dox_pilot.csv')
df.columns = ["wild_saline", "wild_dox"]

# Assuming data from knockout
# Working assumption: there is no drop in fractional shortening
df["knockout_saline"] = df["wild_saline"]
df["knockout_dox"] = df["wild_saline"]  # Assumed no drop

means = df.mean()
variances = df.var()
size = 6 # Number of observations per group
alpha = 0.05 # Significance level
significant = []

# Parametric bootstrap
for i in range(5000):

    # Draw new data from Normal distribution using prior parameters above
    draw_bs = np.random.multivariate_normal(means, np.diag(variances), size=size)
    df_bs  = pd.DataFrame(draw_bs, columns=means.index)

    # Test of differences in differences
    # did = (knockout_saline - knockout_dox) - (wild_saline - wild_dox)
    # Null hypothesis:
    #   H0: did == 0
    df_bs = df_bs.stack().reset_index(1).reset_index(drop=True)
    df_bs.columns = ["x", "y"]
    df_bs["dox"] = df_bs["x"].str.contains("saline").astype(float)
    df_bs["knockout"] = df_bs["x"].str.contains("knockout").astype(float)
    res = ols("y ~ dox + knockout + dox:knockout", df_bs).fit()
    pvalue = res.pvalues["dox:knockout"]
    s = pvalue < alpha
    significant.append(s)

# Check how often we were (correctly) able to reject the hypothesis.
print("Power: ", np.mean(significant))  # ~0.77%
