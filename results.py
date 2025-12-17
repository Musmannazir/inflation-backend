import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# -------------------------------------------------------
# DATA FROM YOUR EXPERIMENTS
# -------------------------------------------------------
results = {
    'Model Version': ['Baseline (Weak)', 'Improved (Production)'],
    'MAE (Error)': [2.3277, 0.0649]
}

df = pd.DataFrame(results)

# -------------------------------------------------------
# PLOT CONFIGURATION (Professional Style)
# -------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")

# Create Bar Plot
barplot = sns.barplot(
    x='Model Version', 
    y='MAE (Error)', 
    data=df, 
    palette=['#FF6B6B', '#4ECDC4']  # Red for bad, Teal for good
)

# Add Labels on top of bars
for p in barplot.patches:
    barplot.annotate(
        format(p.get_height(), '.4f'), 
        (p.get_x() + p.get_width() / 2., p.get_height()), 
        ha = 'center', 
        va = 'center', 
        xytext = (0, 9), 
        textcoords = 'offset points',
        fontweight='bold',
        fontsize=12
    )

# Titles and Labels
plt.title('Model Performance Comparison: Baseline vs. Improved', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Mean Absolute Error (Lower is Better)', fontsize=12)
plt.xlabel('')
plt.ylim(0, 3)  # Set limit slightly higher than max error for visibility

# Save the plot
plt.tight_layout()
plt.savefig('experiment_comparison.png', dpi=300)
print("Graph saved as 'experiment_comparison.png'")
plt.show()