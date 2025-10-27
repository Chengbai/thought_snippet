# notebook visualized distribution for multiple float-type columns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.DataFrame({
    'feature_1': np.random.normal(0, 1, 1000),
    'feature_2': np.random.exponential(2, 1000),
    'feature_3': np.random.uniform(-5, 5, 1000),
    'feature_4': np.random.gamma(2, 2, 1000),
    'category': np.random.choice(['A', 'B'], 1000)  # non-float column
}) 

# Select only int or float columns
numerical_cols = df.select_dtypes(include=['float64', 'float32', "int32", "int64"]).columns.tolist()
print(f"Numerical columns: {numerical_cols}")

# Method 1: Histograms in subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols):
    axes[idx].hist(df[col], bins=50, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Distribution of {col}')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Method 2: KDE plots with seaborn
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols):
    sns.kdeplot(data=df, x=col, fill=True, ax=axes[idx])
    axes[idx].set_title(f'KDE of {col}')
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Method 3: Combined histogram + KDE
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols):
    sns.histplot(data=df, x=col, kde=True, bins=50, ax=axes[idx])
    axes[idx].set_title(f'Histogram + KDE of {col}')
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Method 4: Box plots (great for outlier detection)
fig, ax = plt.subplots(figsize=(12, 6))
df[numerical_cols].boxplot(ax=ax)
ax.set_title('Box Plots of All Float Columns')
ax.set_ylabel('Value')
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Method 5: Violin plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols):
    sns.violinplot(y=df[col], ax=axes[idx])
    axes[idx].set_title(f'Violin Plot of {col}')
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Method 6: All distributions overlayed (useful for comparison)
plt.figure(figsize=(12, 6))
for col in numerical_cols:
    sns.kdeplot(data=df, x=col, label=col, fill=False, linewidth=2)
plt.title('Overlay of All Float Column Distributions')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Bonus: Summary statistics
print("\nSummary Statistics:")
print(df[numerical_cols].describe())
