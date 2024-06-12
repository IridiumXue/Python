import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import levene
from statsmodels.formula.api import ols, glm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from statsmodels.multivariate.manova import MANOVA
import statsmodels.api as sm

litter_data = pd.read_csv('path/litter.csv') 

# Levene's Test for equality of variances
levene_test = levene(*[group['weight'].values for name, group in litter_data.groupby('dose')])

# One-way ANOVA for 'gesttime' by 'dose'
model_gesttime = ols('gesttime ~ C(dose)', data=litter_data).fit()
anova_gesttime = anova_lm(model_gesttime)

# Two-way ANOVA for 'weight' by 'dose' and 'gesttime'
model_weight_dose_gesttime = ols('weight ~ C(dose) * gesttime', data=litter_data).fit()
anova_weight_dose_gesttime = anova_lm(model_weight_dose_gesttime)

# One-way ANOVA for 'weight' with 'gesttime' and 'dose' as predictors

model_weight = ols('weight ~ gesttime + C(dose)', data=litter_data).fit()
anova_weight = anova_lm(model_weight)

# Calculate partial Eta squared
eta_squared = model_weight.ess / (model_weight.ess + model_weight.ssr) 

# Tukey's HSD Post-Hoc Test
mc = MultiComparison(litter_data['weight'], litter_data['dose'])
tukey_result = mc.tukeyhsd()

# ANCOVA
ancova_result = glm('weight ~ gesttime + C(dose)', data=litter_data).fit().summary()

# MANOVA
manova = MANOVA.from_formula('weight + gesttime ~ C(dose)', data=litter_data)
manova_result = manova.mv_test()

# Visualization
sns.set_theme(style="whitegrid") 
fig, axes = plt.subplots(1, 5, figsize=(20, 6), sharey=True)
doses = sorted(litter_data['dose'].unique())
colors = sns.color_palette("husl", len(doses))

for i, (ax, dose) in enumerate(zip(axes, doses)):
    subset = litter_data[litter_data['dose'] == dose]
    sns.scatterplot(x='gesttime', y='weight', data=subset, ax=ax, color=colors[i], label=f'Dose: {dose}')
    sns.regplot(x='gesttime', y='weight', data=subset, ax=ax, color=colors[i], scatter=False, ci=None)
    ax.set_title(f'Dose = {dose}')
    ax.set_xlabel('Gesttime')
    if i == 0:
        ax.set_ylabel('Weight')
    else:
        ax.set_ylabel('')

# Superpose all data in the last subplot
for i, dose in enumerate(doses):
    subset = litter_data[litter_data['dose'] == dose]
    sns.scatterplot(x='gesttime', y='weight', data=subset, ax=axes[-1], color=colors[i], label=f'Dose: {dose}')
    sns.regplot(x='gesttime', y='weight', data=subset, ax=axes[-1], color=colors[i], scatter=False, ci=None, line_kws={"alpha": 0.5})

axes[-1].set_title('Superpose')
axes[-1].set_xlabel('Gesttime')
axes[-1].set_ylabel('')
axes[-1].legend(title='Dose', loc='upper left')
plt.tight_layout()
plt.show()

# Print result
print("Levene's Test Result:", levene_test)
print("ANOVA for 'gesttime' by 'dose':", anova_gesttime)
print("Two-way ANOVA for 'weight' by 'dose' and 'gesttime':", anova_weight_dose_gesttime)
print("ANOVA for 'weight' with 'gesttime' and 'dose':", anova_weight)
print("Partial Eta Squared:", eta_squared)
print("Tukey's HSD Test Result:", tukey_result)
print("ANCOVA Result:", ancova_result)
print("MANOVA Result:", manova_result.summary())
