import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading dataset...")
df = pd.read_csv('icwsmR_final_dataset_share.csv')

df['midnight_wt_total'] = df[['hour_0_wt', 'hour_1_wt', 'hour_2_wt', 'hour_3_wt', 'hour_4_wt', 'hour_5_wt']].sum(axis=1)

sns.set_theme(style="whitegrid")

print("Generating graphs...")

# Plot 1: Class Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='preds_3_label_criteria')
plt.title('Dataset Distribution: Addiction Risk Levels', fontsize=14, fontweight='bold')
plt.ylabel('Number of Users')
plt.xlabel('Addiction Risk Level')
plt.tight_layout()
plt.savefig('eda_1_distribution.png', dpi=300)
plt.close()

# Plot 2: The Midnight Spike
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='preds_3_label_criteria', y='midnight_wt_total')
plt.title('EDA Finding 1: The Midnight Spike', fontsize=14, fontweight='bold')
plt.ylabel('Midnight Watch Time (Minutes)')
plt.xlabel('Addiction Risk Level')
plt.tight_layout()
plt.savefig('eda_2_midnight.png', dpi=300)
plt.close()

# Plot 3: Session Frequency
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='preds_3_label_criteria', y='session_all_per_day')
plt.axhline(30, color='red', linestyle='--', label='Compulsive Threshold (>30)')
plt.title('EDA Finding 2: Session Frequency & Compulsive Checking', fontsize=14, fontweight='bold')
plt.ylabel('Daily App Sessions')
plt.xlabel('Addiction Risk Level')
plt.legend()
plt.tight_layout()
plt.savefig('eda_3_sessions.png', dpi=300)
plt.close()

# Plot 4: Content Breadth
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='preds_3_label_criteria', y='category_count_unique')
plt.title('EDA Finding 3: Content Breadth (The Filter Bubble)', fontsize=14, fontweight='bold')
plt.ylabel('Unique Categories Viewed')
plt.xlabel('Addiction Risk Level')
plt.tight_layout()
plt.savefig('eda_4_categories.png', dpi=300)
plt.close()

print("✅ Real, non-invisible EDA graphs successfully saved!")