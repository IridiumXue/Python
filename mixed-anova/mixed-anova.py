import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('path/CO2.csv')

# Convert 'conc' to a categorical variable
data['conc'] = data['conc'].astype('category')

# Subset the data for 'chilled' treatment
chilled_data = data[data['Treatment'] == 'chilled']

# Fit the mixed effects model considering Plant and conc as random effects
model_formula = 'uptake ~ C(conc) * Type'
mixed_model = mixedlm(model_formula, chilled_data, groups=chilled_data['Plant'], re_formula="~C(conc)")
mixed_results = mixed_model.fit()

# Print the summary of the model
print(mixed_results.summary())

# Calculate and display descriptive statistics
descriptive_stats = chilled_data.groupby(['Type', 'conc']).uptake.agg(['mean', 'std', 'count']).reset_index()
print("\nDescriptive Statistics:")
print(descriptive_stats)

# Extract parameter estimates if needed for manual calculations of F-values
fixed_effects = mixed_results.fe_params
print("\nFixed Effects:")
print(fixed_effects)

# Manually compute effect sizes if required from variance components
var_components = mixed_results.cov_re
total_variance = var_components.sum().sum() + mixed_results.scale  # total variance = random effects + residual
fixed_effect_var = (fixed_effects ** 2).sum()  # simplistic approach for demonstration

print("\nVariance Components:")
print(var_components)
print("\nTotal Variance:", total_variance)
print("Fixed Effect Variance Estimate:", fixed_effect_var)

# Visualize interactions
sns.pointplot(data=chilled_data, x='conc', y='uptake', hue='Type', dodge=True, markers=['o', 's'], capsize=.1,
              palette=['lightcoral', 'lightgreen'])  # Specify light red and light green colors
plt.title('Interaction Plot of Uptake by Concentration and Type')
plt.xlabel('Level of Concentration')
plt.ylabel('Level of Uptake')
plt.show()
