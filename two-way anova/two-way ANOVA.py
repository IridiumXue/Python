import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy.stats import shapiro

# Load the data
df = pd.read_csv('path/ToothGrowth.csv') 

# Convert dose to a categorical variable
df['dose'] = df['dose'].astype('category')  

# Shapiro-Wilk test
print("Detailed Shapiro-Wilk Test Results:")
for (dose, supp), group in df.groupby(['dose', 'supp']):
    stat, p = shapiro(group['len'])
    print(f"Dose: {dose}, Supplement: {supp}")
    print(f"Shapiro-Wilk normality test\nData: {group['len'].tolist()}\nW = {stat:.4f}, p-value = {p:.4f}")
    print("-" * 80)

# Two-way ANOVA
model = ols('len ~ C(dose) * C(supp)', data=df).fit()
anova_results = sm.stats.anova_lm(model, typ=2)
print("\nDetailed ANOVA Results:")
print(anova_results)

# Interaction plot
plt.figure(figsize=(10, 6))
sns.pointplot(data=df, x='dose', y='len', hue='supp', markers=["o", "s"], linestyles=["-", "--"],
              dodge=True, capsize=0.1, errwidth=2, ci=95, palette=["#FF9999", "#9999FF"])
plt.title('Interaction Plot of Tooth Length by Dose and Supplement')
plt.xlabel('Dose (mg)')
plt.ylabel('Mean Tooth Length')
plt.show()

# Estimated Marginal Means
predictions = model.get_prediction(df).summary_frame(alpha=0.05)  # 95% CI
df_with_preds = df.copy()
df_with_preds['predicted_mean'] = predictions['mean']
df_with_preds['ci_lower'] = predictions['obs_ci_lower']
df_with_preds['ci_upper'] = predictions['obs_ci_upper']

print("\nEstimated Marginal Means (EMMs) with 95% Confidence Intervals:")
grouped_preds = df_with_preds.groupby(['supp', 'dose'])
for (supp, dose), group in grouped_preds:
    mean = group['predicted_mean'].mean()
    ci_lower = group['ci_lower'].mean()
    ci_upper = group['ci_upper'].mean()
    print(f"Supplement: {supp}, Dose: {dose}")
    print(f"EMMean = {mean:.2f}, SE = {(ci_upper - ci_lower) / 3.92:.2f}, CI = ({ci_lower:.2f}, {ci_upper:.2f})")
    print("-" * 80)

# Visualization of EMMs with confidence intervals
plt.figure(figsize=(10, 6))
sns.pointplot(data=df_with_preds, x='dose', y='predicted_mean', hue='supp', 
              markers=["o", "s"], linestyles=["-", "--"], dodge=True, capsize=0.1,
              errwidth=2, ci=None, palette=["#FF9999", "#9999FF"])
for i in range(len(df_with_preds)):
    plt.errorbar(x=i % 3 + (0.2 if df_with_preds.iloc[i]['supp'] == 'OJ' else -0.2),
                 y=df_with_preds.iloc[i]['predicted_mean'],
                 yerr=[[df_with_preds.iloc[i]['predicted_mean'] - df_with_preds.iloc[i]['ci_lower']],
                       [df_with_preds.iloc[i]['ci_upper'] - df_with_preds.iloc[i]['predicted_mean']]],
                 fmt='none', ecolor='#FF6666' if df_with_preds.iloc[i]['supp'] == 'OJ' else '#6666FF', capthick=2, alpha=0.5)
plt.title('Estimated Marginal Means of Tooth Length by Dose and Supplement')
plt.xlabel('Dose (mg)')
plt.ylabel('Estimated Tooth Length')
plt.grid(True)
plt.legend(title='Supplement Type')
plt.show()