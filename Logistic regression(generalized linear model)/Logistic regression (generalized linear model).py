import pandas as pd
import numpy as np
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Binomial
import statsmodels.api as sm
from scipy.stats import chi2

# Load data
icu = pd.read_csv('path/ICU.csv')

# Recoding
icu['died'] = np.where(icu['died'] == 'No', 0, 1)
icu['sex'] = np.where(icu['sex'] == 'Female', 0, 1)
icu['coma1'] = np.where(icu['coma'] == 'Stupor', 1, 0)
icu['coma2'] = np.where(icu['coma'] == 'Coma', 1, 0)

# Fit the model
model1 = glm('died ~ age + sex + coma1 + coma2 + systolic', family=Binomial(), data=icu).fit()
print(model1.summary())

# Odds Ratios and 95% CI
params = model1.params
conf = model1.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'OR']
print(np.exp(conf))

# Manual calculation of Likelihood Ratio Test
null_model = glm('died ~ 1', family=Binomial(), data=icu).fit()
lr_stat = 2 * (model1.llf - null_model.llf)
lr_df = model1.df_model - null_model.df_model
lr_pvalue = 1 - chi2.cdf(lr_stat, lr_df)
print("Likelihood Ratio Test for Model 1:")
print("Chi-squared:", lr_stat, "p-value:", lr_pvalue)

# Hosmer-Lemeshow Test Function
def hosmer_lemeshow_test(observed, predicted, groups=10):
    data = pd.DataFrame({'observed': observed, 'predicted': predicted})
    data['group'] = pd.qcut(data['predicted'], groups, duplicates='drop')
    grouped = data.groupby('group')
    observed_sum = grouped['observed'].sum()
    expected_sum = grouped['predicted'].sum()
    observed_count = grouped['observed'].count()
    expected_count = observed_count - expected_sum
    hl_stat = ((observed_sum - expected_sum) ** 2 / (expected_sum * (1 - expected_sum / observed_count))).sum()
    p_value = chi2.sf(hl_stat, groups - 2)
    return hl_stat, p_value

hl_stat, hl_pvalue = hosmer_lemeshow_test(icu['died'], model1.fittedvalues)
print("Hosmer-Lemeshow Test:")
print("Chi-squared:", hl_stat, "p-value:", hl_pvalue)

# Model Comparison using Sequential Likelihood Ratio Tests
model2 = glm('died ~ age + coma2 + systolic', family=Binomial(), data=icu).fit()
lr_stat_model_comp = 2 * (model1.llf - model2.llf)
lr_df_model_comp = model1.df_resid - model2.df_resid
lr_pvalue_model_comp = 1 - chi2.cdf(lr_stat_model_comp, lr_df_model_comp)
print("Model comparison (Model 1 vs Model 2):")
print(f"Chi-squared: {lr_stat_model_comp}, Degrees of freedom: {lr_df_model_comp}, p-value: {lr_pvalue_model_comp}")

# Classification Table
pred_y = model1.predict(icu)  # Removed the 'typ' or any other argument
classification_df = pd.DataFrame({'observed_y': icu['died'], 'predicted_y': np.round(pred_y, 0)})
print(pd.crosstab(classification_df['observed_y'], classification_df['predicted_y']))

# Calculate McFadden's Pseudo R-squared
def calculate_pseudo_r_squared(model, null_model):
    llf_full_model = model.llf  # Log-likelihood of the full model
    llf_null_model = null_model.llf  # Log-likelihood of the null model (intercept only)
    pseudo_r_squared = 1 - (llf_full_model / llf_null_model)
    return pseudo_r_squared

# Pseudo R-squared
mcFadden_pseudo_r_squared = calculate_pseudo_r_squared(model1, null_model)
print("McFadden's Pseudo R-squared for Model 1:", mcFadden_pseudo_r_squared)