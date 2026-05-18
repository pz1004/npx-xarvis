"""Generate publication-quality plots for experiment results."""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load results
with open("experiment_results_v2.json") as f:
    results = json.load(f)

method_names = ["Proposed CONF", "Proposed POL", "Proposed REF", "BA Filter (baseline)"]
noise_types = ["ba", "shot", "mixed"]
ratios = [0.5, 1.0, 2.0, 5.0, 10.0]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
markers = ['o', 's', '^', 'D']

# ============================================================================
# Plot 1: AUC vs Noise Ratio for each noise type
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

for ax_idx, noise_type in enumerate(noise_types):
    ax = axes[ax_idx]
    for m_idx, method_name in enumerate(method_names):
        aucs = []
        for ratio in ratios:
            r = [x for x in results if x["method"] == method_name 
                 and x["noise_type"] == noise_type and x["ratio"] == ratio]
            aucs.append(r[0]["auc"] if r else 0.5)
        ax.plot(ratios, aucs, marker=markers[m_idx], color=colors[m_idx],
                linewidth=2, markersize=7, label=method_name)
    
    ax.set_xlabel("Noise Ratio", fontsize=11)
    ax.set_title(f"{noise_type.upper()} Noise", fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xticks(ratios)
    ax.set_xticklabels([str(r) for r in ratios])
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)

axes[0].set_ylabel("AUC", fontsize=11)
axes[2].legend(loc='lower left', fontsize=9)
plt.suptitle("Noise Discrimination Performance (AUC) vs Noise Ratio", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("fig_auc_vs_noise_ratio.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_auc_vs_noise_ratio.png")

# ============================================================================
# Plot 2: ESR vs Noise Ratio
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

for ax_idx, noise_type in enumerate(noise_types):
    ax = axes[ax_idx]
    for m_idx, method_name in enumerate(method_names):
        esrs = []
        for ratio in ratios:
            r = [x for x in results if x["method"] == method_name 
                 and x["noise_type"] == noise_type and x["ratio"] == ratio]
            esrs.append(r[0]["esr"] if r else 0.0)
        ax.plot(ratios, esrs, marker=markers[m_idx], color=colors[m_idx],
                linewidth=2, markersize=7, label=method_name)
    
    ax.set_xlabel("Noise Ratio", fontsize=11)
    ax.set_title(f"{noise_type.upper()} Noise", fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xticks(ratios)
    ax.set_xticklabels([str(r) for r in ratios])
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.2, 1.0)

axes[0].set_ylabel("ESR (Event Structural Ratio)", fontsize=11)
axes[2].legend(loc='lower left', fontsize=9)
plt.suptitle("Structural Preservation (ESR) vs Noise Ratio", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("fig_esr_vs_noise_ratio.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_esr_vs_noise_ratio.png")

# ============================================================================
# Plot 3: Aggregate bar chart
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(method_names))
width = 0.25

mean_aucs = []
mean_esrs = []
mean_crs = []
for method_name in method_names:
    mr = [r for r in results if r["method"] == method_name]
    mean_aucs.append(np.mean([r["auc"] for r in mr]))
    mean_esrs.append(np.mean([r["esr"] for r in mr]))
    mean_crs.append(np.mean([1.0/r["cr"] for r in mr]))  # Inverse CR = filtering ratio

bars1 = ax.bar(x - width, mean_aucs, width, label='AUC', color='#2196F3', alpha=0.8)
bars2 = ax.bar(x, mean_esrs, width, label='ESR', color='#4CAF50', alpha=0.8)
bars3 = ax.bar(x + width, mean_crs, width, label='1/CR (Keep Ratio)', color='#FF9800', alpha=0.8)

ax.set_xlabel('Method', fontsize=11)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('Aggregate Performance Comparison', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(method_names, fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("fig_aggregate_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_aggregate_comparison.png")

# ============================================================================
# Plot 4: Speedup visualization
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 4))

categories = ['Python (Baseline)', 'Numba JIT (Optimized)']
throughputs = [56685, 5018631]  # From experiment v1
speedup = throughputs[1] / throughputs[0]

bars = ax.barh(categories, throughputs, color=['#FF5722', '#4CAF50'], height=0.5)
ax.set_xlabel('Throughput (events/second)', fontsize=11)
ax.set_title(f'Filter Processing Throughput (Speedup: {speedup:.1f}x)', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3, axis='x')

for bar, val in zip(bars, throughputs):
    ax.text(val * 1.1, bar.get_y() + bar.get_height()/2, 
            f'{val:,.0f} eps', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("fig_throughput_speedup.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_throughput_speedup.png")

print("\nAll plots generated successfully!")
