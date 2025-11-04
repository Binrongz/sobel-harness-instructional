import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300

print("Starting to generate all plots...")
print("="*60)

# ==================== Load Data ====================

# CPU data
cpu_threads = [1, 2, 4, 8, 16]
cpu_runtime = [0.263923, 0.131991, 0.0692708, 0.0384475, 0.0207072]
cpu_speedup = [cpu_runtime[0]/r for r in cpu_runtime]

# Load CUDA runtime data
cuda_runtime_df = pd.read_csv('all_cuda_results.txt')
print(f"\n✓ Loaded CUDA runtime data: {len(cuda_runtime_df)} configurations")

# Load CUDA NCU data
cuda_ncu_df = pd.read_csv('complete_ncu_data.csv')
print(f"✓ Loaded CUDA NCU data: {len(cuda_ncu_df)} configurations")

# Prepare data for heatmaps
blocks_list = [1, 4, 16, 64, 256, 1024, 4096]
threads_list = [32, 64, 128, 256, 512, 1024]

# ==================== Figure 1: CPU Scaling ====================
print("\nGenerating Figure 1: CPU Scaling...")

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(cpu_threads, cpu_runtime, marker='o', linewidth=2.5, 
        markersize=10, color='#2E86AB', label='Runtime')

ax.set_xlabel('Number of Threads', fontsize=14, fontweight='bold')
ax.set_ylabel('Runtime (seconds)', fontsize=14, fontweight='bold')
ax.set_title('CPU Performance Scaling with OpenMP', fontsize=16, fontweight='bold')
ax.set_xticks(cpu_threads)
ax.grid(True, alpha=0.3)

# Add value labels
for t, r in zip(cpu_threads, cpu_runtime):
    ax.text(t, r*1.05, f'{r:.3f}', ha='center', va='bottom', 
            fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('figure1_cpu_scaling.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figure1_cpu_scaling.png")

# ==================== Figure 2: CPU Speedup ====================
print("Generating Figure 2: CPU Speedup...")

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(cpu_threads, cpu_speedup, marker='o', linewidth=2.5, 
        markersize=10, color='#A23B72', label='Actual Speedup')
ax.plot(cpu_threads, cpu_threads, linestyle='--', linewidth=2, 
        color='#F18F01', label='Ideal Speedup (Linear)', alpha=0.8)

ax.set_xlabel('Number of Threads', fontsize=14, fontweight='bold')
ax.set_ylabel('Speedup', fontsize=14, fontweight='bold')
ax.set_title('CPU Speedup Relative to Single Thread', fontsize=16, fontweight='bold')
ax.set_xticks(cpu_threads)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12, loc='upper left')

# Add value labels
for t, s in zip(cpu_threads, cpu_speedup):
    ax.text(t, s*1.03, f'{s:.2f}x', ha='center', va='bottom', 
            fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('figure2_cpu_speedup.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figure2_cpu_speedup.png")

# ==================== Figure 3: CUDA Runtime Heatmap ====================
print("Generating Figure 3: CUDA Runtime Heatmap...")

# Create runtime matrix
runtime_matrix = np.zeros((len(threads_list), len(blocks_list)))
for _, row in cuda_runtime_df.iterrows():
    if row['Blocks'] in blocks_list and row['Threads'] in threads_list:
        i = threads_list.index(row['Threads'])
        j = blocks_list.index(row['Blocks'])
        runtime_matrix[i, j] = row['Runtime']

fig, ax = plt.subplots(figsize=(14, 8))

# Create heatmap
im = ax.imshow(runtime_matrix, cmap='YlOrRd_r', aspect='auto', 
               interpolation='nearest')

# Set ticks
ax.set_xticks(range(len(blocks_list)))
ax.set_yticks(range(len(threads_list)))
ax.set_xticklabels(blocks_list, fontsize=12)
ax.set_yticklabels(threads_list, fontsize=12)

# Labels
ax.set_xlabel('Number of Thread Blocks', fontsize=14, fontweight='bold')
ax.set_ylabel('Threads Per Block', fontsize=14, fontweight='bold')
ax.set_title('CUDA Performance: Runtime (seconds)', fontsize=16, fontweight='bold', pad=20)

# Add text annotations
for i in range(len(threads_list)):
    for j in range(len(blocks_list)):
        value = runtime_matrix[i, j]
        if value > 0:
            text_color = 'white' if value > 0.15 else 'black'
            ax.text(j, i, f'{value:.4f}', ha='center', va='center',
                   color=text_color, fontsize=8, fontweight='bold')

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Runtime (seconds)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('figure3_cuda_runtime_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figure3_cuda_runtime_heatmap.png")

# ==================== Figure 4: CUDA Occupancy Heatmap ====================
print("Generating Figure 4: CUDA Occupancy Heatmap...")

# Create occupancy matrix
occupancy_matrix = np.full((len(threads_list), len(blocks_list)), np.nan)
for _, row in cuda_ncu_df.iterrows():
    if row['Blocks'] in blocks_list and row['Threads'] in threads_list:
        i = threads_list.index(row['Threads'])
        j = blocks_list.index(row['Blocks'])
        occupancy_matrix[i, j] = row['Occupancy']

fig, ax = plt.subplots(figsize=(14, 8))

# Create masked array for NaN values
masked_data = np.ma.masked_where(np.isnan(occupancy_matrix), occupancy_matrix)

# Create heatmap
im = ax.imshow(masked_data, cmap='RdYlGn', aspect='auto', 
               vmin=0, vmax=100, interpolation='nearest')

# Set ticks
ax.set_xticks(range(len(blocks_list)))
ax.set_yticks(range(len(threads_list)))
ax.set_xticklabels(blocks_list, fontsize=12)
ax.set_yticklabels(threads_list, fontsize=12)

# Labels
ax.set_xlabel('Number of Thread Blocks', fontsize=14, fontweight='bold')
ax.set_ylabel('Threads Per Block', fontsize=14, fontweight='bold')
ax.set_title('CUDA Performance: Achieved Occupancy (%)', fontsize=16, fontweight='bold', pad=20)

# Add text annotations
for i in range(len(threads_list)):
    for j in range(len(blocks_list)):
        value = occupancy_matrix[i, j]
        if not np.isnan(value):
            ax.text(j, i, f'{value:.1f}', ha='center', va='center',
                   color='black', fontsize=9, fontweight='bold')

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Occupancy (%)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('figure4_cuda_occupancy_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figure4_cuda_occupancy_heatmap.png")

# ==================== Figure 5: CUDA Bandwidth Heatmap ====================
print("Generating Figure 5: CUDA Bandwidth Heatmap...")

# Create bandwidth matrix
bandwidth_matrix = np.full((len(threads_list), len(blocks_list)), np.nan)
for _, row in cuda_ncu_df.iterrows():
    if row['Blocks'] in blocks_list and row['Threads'] in threads_list:
        i = threads_list.index(row['Threads'])
        j = blocks_list.index(row['Blocks'])
        bandwidth_matrix[i, j] = row['Bandwidth']

fig, ax = plt.subplots(figsize=(14, 8))

# Create masked array for NaN values
masked_data = np.ma.masked_where(np.isnan(bandwidth_matrix), bandwidth_matrix)

# Create heatmap
im = ax.imshow(masked_data, cmap='Blues', aspect='auto', 
               vmin=0, vmax=100, interpolation='nearest')

# Set ticks
ax.set_xticks(range(len(blocks_list)))
ax.set_yticks(range(len(threads_list)))
ax.set_xticklabels(blocks_list, fontsize=12)
ax.set_yticklabels(threads_list, fontsize=12)

# Labels
ax.set_xlabel('Number of Thread Blocks', fontsize=14, fontweight='bold')
ax.set_ylabel('Threads Per Block', fontsize=14, fontweight='bold')
ax.set_title('CUDA Performance: Memory Bandwidth Utilization (%)', 
             fontsize=16, fontweight='bold', pad=20)

# Add text annotations
for i in range(len(threads_list)):
    for j in range(len(blocks_list)):
        value = bandwidth_matrix[i, j]
        if not np.isnan(value):
            ax.text(j, i, f'{value:.1f}', ha='center', va='center',
                   color='black', fontsize=9, fontweight='bold')

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Bandwidth Utilization (%)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('figure5_cuda_bandwidth_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figure5_cuda_bandwidth_heatmap.png")

# ==================== Figure 6: Method Comparison ====================
print("Generating Figure 6: Method Comparison...")

methods = ['CPU\n(1 thread)', 'CPU\n(16 threads)', 
           'CUDA\nBest Config', 'OpenMP\nOffload']
runtimes = [0.263923, 0.0207072, 0.00799659, 2.40924]
colors = ['#89CFF0', '#90EE90', '#FFD700', '#FF6B6B']

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.bar(methods, runtimes, color=colors, edgecolor='black', linewidth=2)

ax.set_ylabel('Runtime (seconds)', fontsize=14, fontweight='bold')
ax.set_title('Performance Comparison: CPU vs GPU Implementations', 
             fontsize=16, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, r in zip(bars, runtimes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height*1.3,
            f'{r:.4f}s',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('figure6_method_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figure6_method_comparison.png")

# ==================== Summary ====================
print("\n" + "="*60)
print("✓ ALL FIGURES GENERATED SUCCESSFULLY!")
print("="*60)
print("\nGenerated files:")
print("  1. figure1_cpu_scaling.png")
print("  2. figure2_cpu_speedup.png")
print("  3. figure3_cuda_runtime_heatmap.png")
print("  4. figure4_cuda_occupancy_heatmap.png")
print("  5. figure5_cuda_bandwidth_heatmap.png")
print("  6. figure6_method_comparison.png")
print("\nData statistics:")
print(f"  - CPU configurations tested: {len(cpu_threads)}")
print(f"  - CUDA runtime configurations: {len(cuda_runtime_df)}")
print(f"  - CUDA NCU configurations: {len(cuda_ncu_df)}")
print(f"  - Best CUDA config: {cuda_runtime_df.loc[cuda_runtime_df['Runtime'].idxmin(), 'Blocks']:.0f} blocks × {cuda_runtime_df.loc[cuda_runtime_df['Runtime'].idxmin(), 'Threads']:.0f} threads")
print(f"  - Best CUDA runtime: {cuda_runtime_df['Runtime'].min():.6f} seconds")
print("\n" + "="*60)