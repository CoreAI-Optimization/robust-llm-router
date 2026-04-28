import matplotlib.pyplot as plt


# Create comprehensive comparison plots with baselines
def plot_csv_comparison_with_baselines(loaded_results, baseline_df, gpu_counts,
                                       output_filename='optimization_comparison_from_csv.png',
                                       baseline_performance_threshold=None, test_name=None):
    """
    Plot comparison of optimization results from multiple CSV files with baselines.
    
    Parameters
    ----------
    loaded_results : dict
        Dictionary with model names as keys, each containing 'df' and 'max_cost'
    baseline_df : pd.DataFrame
        DataFrame of baseline results with columns: llm_name, gpu_budget, test_performance,
        cost, gpus_used, small_instances, large_instances, opensource_instances, closedsource_instances
    gpu_counts : list
        List of GPU counts corresponding to baseline data points
    output_filename : str
        Filename for saved plot
    baseline_performance_threshold : float, optional
        If set, only include baselines whose test_performance is >= this value.
    """
    # Golden ratio: 1.618
    golden_ratio = 1.618
    fig_width = 16
    fig_height = fig_width / golden_ratio
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height))
    axes = axes.flatten()
    
    baseline_colors = ['#CD5C5C', '#5F9EA0', '#DAA520', '#9370DB', '#BC8F8F', '#6B8E23', '#B8860B', '#4682B4']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    if test_name == "rb":
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        

    # Plot each loaded result
    for idx, (label, data) in enumerate(loaded_results.items()):
        df = data['df']
        max_cost = data['max_cost']
        color = colors[idx % len(colors)]
        
        # Remove NaN rows
        valid_df = df.dropna(subset=['test_performance'])
        
        if len(valid_df) == 0:
            continue
        
        # Plot 0: Test Performance vs GPU Budget
        axes[0].plot(valid_df['gpu_budget'], valid_df['test_performance'], 
                    f'-o', color=color, linewidth=2.5, markersize=8, 
                    label=f'{label} (Cost={max_cost:.1e})', alpha=0.85)
        
        # Plot 1: Open-source instances only
        axes[1].plot(valid_df['gpu_budget'], valid_df['opensource_instances'], 
                    f'-o', color=color, linewidth=2, markersize=6, 
                    label=f'{label}', alpha=0.75)
        
        # Plot 2: Cost per Query
        axes[2].plot(valid_df['gpu_budget'], valid_df['avg_cost_per_query'], 
                    f'-o', color=color, linewidth=2.5, markersize=8, 
                    label=label, alpha=0.85)
        
        # Plot 3: Small vs Large instances
        axes[3].plot(valid_df['gpu_budget'], valid_df['small_instances'], 
                    f'-o', color=color, linewidth=2, markersize=6, 
                    label=f'{label} (Small)', alpha=0.75)
        axes[3].plot(valid_df['gpu_budget'], valid_df['large_instances'], 
                    f'--s', color=color, linewidth=2, markersize=6, 
                    label=f'{label} (Large)', alpha=0.75)
    
    # Add baselines if provided
    if baseline_df is not None and len(baseline_df) > 0:
        baseline_df = baseline_df.dropna(subset=['test_performance']).copy()
        if baseline_performance_threshold is not None:
            baseline_df = baseline_df[baseline_df['test_performance'] >= baseline_performance_threshold]
        if len(baseline_df) == 0:
            baseline_df = None
    if baseline_df is not None and len(baseline_df) > 0:
        # Calculate offset pattern for better visibility of overlapping points
        unique_baselines = baseline_df['llm_name'].unique()
        num_baselines = len(unique_baselines)
        offset_step = 15  # GPU budget offset step
        
        for baseline_idx, baseline_name in enumerate(unique_baselines):
            baseline_rows = baseline_df[baseline_df['llm_name'] == baseline_name].dropna(subset=['test_performance'])
            if len(baseline_rows) == 0:
                continue
            label_name = baseline_name.split('/', 1)[-1]
            row = baseline_rows.iloc[0]
            baseline_color = baseline_colors[baseline_idx % len(baseline_colors)]
            
            gpu_val = row['gpu_budget']
            perf_test = row['test_performance']
            cost_val = row['cost']
            gpu_used = row['gpus_used']
            small_i = row['small_instances']
            large_i = row['large_instances']
            open_i = row['opensource_instances']
            closed_i = row['closedsource_instances']

            # Add horizontal offset to spread out overlapping points
            gpu_offset = (baseline_idx - num_baselines/2) * offset_step
            gpu_val_offset = gpu_val + gpu_offset

            # Plot 0: Test Performance
            axes[0].scatter(gpu_val_offset, perf_test, color=baseline_color, marker='D', s=180,
                          alpha=0.9, edgecolors='black', linewidths=1.5, zorder=10 + baseline_idx,
                          label=f'Baseline: {label_name}')

            # Plot 1: Open-source instances only
            if open_i > 0:
                axes[1].scatter(gpu_val_offset, open_i, color=baseline_color, marker='D', s=150,
                              alpha=0.9, edgecolors='black', linewidths=1.5, zorder=10 + baseline_idx,
                              label=f'Baseline: {label_name}')

            # Plot 2: Cost
            axes[2].scatter(gpu_val_offset, cost_val, color=baseline_color, marker='D', s=180,
                          alpha=0.9, edgecolors='black', linewidths=1.5, zorder=10 + baseline_idx,
                          label=f'Baseline: {label_name}')

            # Plot 3: Small/Large instances (only open-source baselines)
            # Check if baseline is open-source
            is_opensource = baseline_name.lower() not in {
                            'gpt_4o',
                            'glm_4_plus',
                            'claude35_haiku20241022',
                            'gpt_4o_mini',
                            'gpt_4o_mini_cot',
                            'gemini15_flash'
                        }

            # if is_opensource:
                #     if small_i > 0:
                #         axes[3].scatter(gpu_val_offset, small_i, color=baseline_color, marker='D', s=150, 
                #                       alpha=0.9, edgecolors='black', linewidths=1.5, zorder=10 + baseline_idx,
                #                       label=f'Baseline: {baseline_name}')
                #     if large_i > 0:
                #         axes[3].scatter(gpu_val_offset, large_i, color=baseline_color, marker='s', s=130, 
                #                       alpha=0.8, edgecolors='black', linewidths=1.5, zorder=10 + baseline_idx)
    
    # Configure all subplots
    axes[0].set_xlabel('GPU Budget', fontsize=13)
    axes[0].set_ylabel('Test Performance', fontsize=13)
    axes[0].set_title('Model Routing: Full Optimization vs Baselines', fontsize=15, fontweight='bold')
    axes[0].legend(fontsize=9, loc='best', framealpha=0.95)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].tick_params(axis='both', which='major', labelsize=11)
    
    axes[1].set_xlabel('GPU Budget', fontsize=13)
    axes[1].set_ylabel('Number of Instances', fontsize=13)
    axes[1].set_title('Open-Source Model Instances', fontsize=15, fontweight='bold')
    axes[1].legend(fontsize=8, loc='best', framealpha=0.95)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].tick_params(axis='both', which='major', labelsize=11)
    
    axes[2].set_xlabel('GPU Budget', fontsize=13)
    axes[2].set_ylabel('Average Cost per Query', fontsize=13)
    axes[2].set_title('Cost vs GPU Budget', fontsize=15, fontweight='bold')
    axes[2].legend(fontsize=9, loc='best', framealpha=0.95)
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    axes[2].tick_params(axis='both', which='major', labelsize=11)
    
    axes[3].set_xlabel('GPU Budget', fontsize=13)
    axes[3].set_ylabel('Number of Instances', fontsize=13)
    axes[3].set_title('Small vs Large Open-Source Model Instances', fontsize=15, fontweight='bold')
    axes[3].legend(fontsize=8, loc='best', framealpha=0.2)
    axes[3].grid(True, alpha=0.3, linestyle='--')
    axes[3].tick_params(axis='both', which='major', labelsize=11)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=200, bbox_inches='tight')
    plt.show()
    
    print(f"\nComparison plot saved as '{output_filename}'")
    
    return fig, axes


# Create focused single-plot comparison (Test Performance only)
def plot_csv_focused_comparison(loaded_results, baseline_df, gpu_counts,
                                output_filename='optimization_comparison_focused.png',
                                baseline_performance_threshold=None, test_name=None):
    """
    Create a focused single-plot comparison showing only test performance.
    
    Parameters
    ----------
    loaded_results : dict
        Dictionary with model names as keys, each containing 'df' and 'max_cost'
    baseline_df : pd.DataFrame
        DataFrame of baseline results with columns: llm_name, gpu_budget, test_performance
    gpu_counts : list
        List of GPU counts corresponding to baseline data points
    output_filename : str
        Filename for saved plot
    baseline_performance_threshold : float, optional
        If set, only include baselines whose test_performance is >= this value.
    """
    # Golden ratio: 1.618
    golden_ratio = 1.618
    height = 7
    width = height * golden_ratio

    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    baseline_colors = ['#CD5C5C', '#5F9EA0', '#DAA520', '#9370DB', '#BC8F8F', '#6B8E23', '#B8860B', '#4682B4']
    
    # if test_name == "rb":
    #     colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Plot optimization results
    for idx, (label, data) in enumerate(loaded_results.items()):
        df = data['df']
        max_cost = data['max_cost']
        color = colors[idx % len(colors)]
        
        valid_df = df.dropna(subset=['test_performance'])
        
        if len(valid_df) > 0:
            ax.plot(valid_df['gpu_budget'], valid_df['test_performance'], 
                   f'-o', color=color, linewidth=3, markersize=10, 
                   label=f'{label}', alpha=0.85)
    
    # Add baselines
    if baseline_df is not None and len(baseline_df) > 0:
        baseline_df = baseline_df.dropna(subset=['test_performance']).copy()
        if baseline_performance_threshold is not None:
            baseline_df = baseline_df[baseline_df['test_performance'] >= baseline_performance_threshold]
        if len(baseline_df) == 0:
            baseline_df = None
    if baseline_df is not None and len(baseline_df) > 0:
        unique_baselines = baseline_df['llm_name'].unique()
        for baseline_idx, baseline_name in enumerate(unique_baselines):
            baseline_rows = baseline_df[baseline_df['llm_name'] == baseline_name].dropna(subset=['test_performance'])
            if len(baseline_rows) == 0:
                continue
            row = baseline_rows.iloc[0]
            baseline_color = baseline_colors[baseline_idx % len(baseline_colors)]
            
            gpu_val = row['gpu_budget']
            perf_test = row['test_performance']

            ax.scatter(gpu_val, perf_test, color=baseline_color, marker='D', s=250,
                      alpha=0.95, edgecolors='black', linewidths=2, zorder=10)

            y_offset = -0.005
            x_offset = -100 if test_name == 'test1' else 0
            if baseline_name == 'deepseek_coder':
                y_offset = -0.01
            if baseline_name == 'llama31_405b_instruct':
                x_offset = -200
            if baseline_name == 'gemini15_flash':
                y_offset = 0
                x_offset = 40
            
            label_name = baseline_name.split('/', 1)[-1]
            ax.text(gpu_val + x_offset, perf_test + y_offset, f' {label_name}',
                   fontsize=12, fontweight='bold',
                   verticalalignment='center', color=baseline_color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            alpha=0.3, edgecolor=baseline_color, linewidth=1.5))
    
    ax.set_xlabel('GPU Budget', fontsize=20)
    ax.set_ylabel('Test Performance', fontsize=20)
    ax.set_title('Model Routing: Full Optimization vs Baselines', 
                fontsize=20, fontweight='bold')
    ax.legend(fontsize=18, loc='lower right', framealpha=0.3)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Adjust margins to use full space
    plt.tight_layout(pad=1.5)
    plt.savefig(output_filename, dpi=200, bbox_inches='tight', pad_inches=0.2)
    plt.show()
    
    print(f"\nFocused comparison plot saved as '{output_filename}'")
    
    return fig, ax
