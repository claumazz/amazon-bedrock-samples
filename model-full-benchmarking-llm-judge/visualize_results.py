import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
from matplotlib.backends.backend_pdf import PdfPages
import glob



# ----------------------------------------
# VISUALIZATION FUNCTIONS
# ----------------------------------------
def create_integrated_analysis_charts(df, performance_metrics, latency_metrics, pdf):
    """Create charts combining performance, latency and cost metrics"""
    # Merge performance and latency metrics
    perf_df = performance_metrics.reset_index()
    latency_df = latency_metrics.reset_index()

    # Ensure they have the same index
    integrated_df = pd.merge(
        perf_df,
        latency_df[['model', 'inference_profile', 'TTFT_mean', 'OTPS_mean']],
        on=['model', 'inference_profile']
    )

    # Check which success columns are available
    success_col = 'evaluation_success_rate'

    # Create model labels
    integrated_df['model_label'] = integrated_df.apply(
        lambda x: f"{x['model'].split('.')[-1]}\n({x['inference_profile']})", axis=1
    )

    # 1. Create 3D-like visualization (Success vs TTFT vs Cost)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)

    # Color represents cost (darker = more expensive)
    scatter = ax.scatter(
        integrated_df['TTFT_mean'],
        integrated_df[success_col],
        s=integrated_df['OTPS_mean'] * 10,  # Scale for visibility
        c=integrated_df['avg_cost_per_response'],
        cmap='coolwarm_r',  # Reversed so blue = cheaper
        alpha=0.7
    )

    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cost per Response ($)')

    # Add model labels to points
    for i, row in integrated_df.iterrows():
        ax.annotate(
            row['model_label'],
            (row['TTFT_mean'], row[success_col]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=9
        )

    # Mark the "optimal" region
    optimal_x = min(integrated_df['TTFT_mean']) * 1.5
    optimal_y = 0.9

    # Shade the optimal region
    rect = plt.Rectangle(
        (0, optimal_y), optimal_x, 0.1,
        alpha=0.1, color='green'
    )
    ax.add_patch(rect)
    ax.text(
        optimal_x/2, optimal_y + 0.05,
        "Optimal Region",
        ha='center', fontsize=10
    )

    plt.title('Integrated Analysis: Performance vs Speed vs Cost', fontsize=14)
    plt.xlabel('Time to First Token (seconds)', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)

    # Add legend for circle size
    handles, labels = [], []
    for otps in [10, 20, 30, 40]:
        handles.append(plt.scatter([], [], s=otps*10, color='gray', alpha=0.7))
        labels.append(f'{otps} tokens/sec')

    plt.legend(
        handles, labels,
        title="Output Speed (OTPS)",
        loc='lower right',
        frameon=True,
        framealpha=0.7
    )

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # 2. Radar chart showing multiple dimensions
    # Normalize all metrics to 0-1 scale for radar chart
    metrics_to_include = [
        success_col,
        'value_ratio',
        'TTFT_mean',
        'OTPS_mean',
        'avg_cost_per_response'
    ]

    # Create a copy for normalization
    radar_df = integrated_df.copy()

    # Normalize each metric (0-1 scale)
    for metric in metrics_to_include:
        if metric in ['TTFT_mean', 'avg_cost_per_response']:
            # For these metrics, lower is better, so invert
            max_val = radar_df[metric].max()
            min_val = radar_df[metric].min()
            if max_val > min_val:  # Avoid division by zero
                radar_df[f'norm_{metric}'] = 1 - ((radar_df[metric] - min_val) / (max_val - min_val))
            else:
                radar_df[f'norm_{metric}'] = 1.0
        else:
            # For these metrics, higher is better
            max_val = radar_df[metric].max()
            min_val = radar_df[metric].min()
            if max_val > min_val:  # Avoid division by zero
                radar_df[f'norm_{metric}'] = (radar_df[metric] - min_val) / (max_val - min_val)
            else:
                radar_df[f'norm_{metric}'] = 1.0

    # Set up radar chart (as many models as will fit well on a page)
    max_models_per_chart = 4
    models_to_plot = list(radar_df['model_label'].unique())

    # Create multiple radar charts if needed
    for chart_idx in range(0, len(models_to_plot), max_models_per_chart):
        models_subset = models_to_plot[chart_idx:chart_idx+max_models_per_chart]

        # Set up radar chart
        metric_labels = [
            'Success Rate',
            'Value Ratio',
            'Response Speed',  # Normalized TTFT
            'Generation Speed',  # OTPS
            'Cost Efficiency'   # Inverted cost
        ]

        num_metrics = len(metric_labels)
        angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the circle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        # Plot each model
        for model_label in models_subset:
            model_data = radar_df[radar_df['model_label'] == model_label]

            # Get normalized values
            values = [
                model_data[f'norm_{success_col}'].values[0],
                model_data['norm_value_ratio'].values[0],
                model_data['norm_TTFT_mean'].values[0],
                model_data['norm_OTPS_mean'].values[0],
                model_data['norm_avg_cost_per_response'].values[0]
            ]
            values += values[:1]  # Close the circle

            # Plot the model on the radar chart
            ax.plot(angles, values, linewidth=2, label=model_label)
            ax.fill(angles, values, alpha=0.1)

        # Set chart properties
        ax.set_theta_offset(np.pi / 2)  # Start from top
        ax.set_theta_direction(-1)  # Clockwise

        # Draw labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)

        # Draw y-axis labels (0-1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.set_ylim(0, 1)

        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        plt.title(f'Multi-Dimensional Model Comparison (Chart {chart_idx//max_models_per_chart + 1})', fontsize=14)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()


def create_comparison_charts(df, pdf):
    """Create charts comparing model performance on all tasks"""
    plt.figure(figsize=(14, 8))

    # Get unique models and profiles for comparison
    models = df['model'].unique()
    comparison_df = df.groupby(['model', 'inference_profile']).agg({
        'task_success': 'mean',
        # 'judge_success': 'mean'
    }).reset_index()
    comparison_df.columns = ['model', 'inference_profile', 'Judge Evaluation']
    # comparison_df.columns = ['model', 'inference_profile', 'Algorithm Metrics', 'Judge Evaluation']

    # Create model labels for x-axis
    comparison_df['model_label'] = comparison_df.apply(
        lambda x: f"{x['model'].split('.')[-1]}\n({x['inference_profile']})", axis=1
    )

    # Reshape for better plotting
    melted_df = pd.melt(
        comparison_df,
        id_vars=['model_label'],
        value_vars=[col for col in comparison_df.columns if col not in ['model', 'inference_profile', 'model_label']],
        var_name='Evaluation Method',
        value_name='Success Rate'
    )

    # Create the comparison chart
    plt.figure(figsize=(14, 8))
    sns.barplot(data=melted_df, x='model_label', y='Success Rate', hue='Evaluation Method')
    plt.title('Model Performance Comparison: Success Rate by Evaluation Method', fontsize=14)
    plt.ylabel('Success Rate', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # 2. Create task type specific comparison
    all_task_types = list(set(df['task_types'].tolist()))
    # For each task type, create a comparison
    for task in all_task_types:
        # Filter data to only include this task type
        task_df = df[df['task_types'].str.contains(task, na=False)]
        # df['judge_explanation']
        if len(task_df) > 0:
            # Group by model and profile
            task_comparison = task_df.groupby(['model', 'inference_profile']).agg({
                'task_success': 'mean'
            }).reset_index()

            # Create model labels
            task_comparison['model_label'] = task_comparison.apply(
                lambda x: f"{x['model'].split('.')[-1]}\n({x['inference_profile']})", axis=1
            )

            # Create the chart
            plt.figure(figsize=(14, 8))
            sns.barplot(data=task_comparison, x='model_label', y='task_success')
            plt.title(f'Model Performance Comparison: {task} Task Success Rate', fontsize=14)
            plt.ylabel('Success Rate', fontsize=12)
            plt.xlabel('Model', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.ylim(0, 1.05)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

def create_latency_comparison_charts(df, latency_metrics, pdf):
    """Create comprehensive latency comparison charts"""
    # 1. TTFT comparison (bar chart)
    plot_df = latency_metrics.reset_index()
    plot_df['model_label'] = plot_df.apply(
        lambda x: f"{x['model'].split('.')[-1]}\n({x['inference_profile']})", axis=1
    )

    plt.figure(figsize=(14, 8))
    ax = sns.barplot(data=plot_df, x='model_label', y='TTFT_mean')

    # Add P90 error bars
    for i, row in plot_df.iterrows():
        plt.errorbar(i, row['TTFT_mean'], yerr=row['TTFT_p90' ] -row['TTFT_mean'],
                     fmt='o', color='black', capsize=5)

    plt.title('Time to First Token (TTFT) Comparison', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels on bars
    for i, v in enumerate(plot_df['TTFT_mean']):
        ax.text(i, v + 0.05, f"{v:.2f}s", ha='center', fontsize=9)

    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # 2. OTPS comparison (bar chart)
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(data=plot_df, x='model_label', y='OTPS_mean')

    plt.title('Output Tokens Per Second (OTPS) Comparison', fontsize=14)
    plt.ylabel('Tokens per Second', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels on bars
    for i, v in enumerate(plot_df['OTPS_mean']):
        ax.text(i, v + 0.5, f"{v:.1f}", ha='center', fontsize=9)

    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # 3. TTFT vs OTPS scatter plot
    plt.figure(figsize=(12, 10))

    # Create scatter plot
    scatter = plt.scatter(
        plot_df['TTFT_mean'],
        plot_df['OTPS_mean'],
        s=200,
        c=plot_df.index,
        cmap='viridis',
        alpha=0.7
    )

    # Add model labels to points
    for i, row in plot_df.iterrows():
        plt.annotate(
            row['model_label'],
            (row['TTFT_mean'], row['OTPS_mean']),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=9
        )

    plt.title('Latency Tradeoff: TTFT vs OTPS', fontsize=14)
    plt.xlabel('Time to First Token (seconds)', fontsize=12)
    plt.ylabel('Output Tokens Per Second', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add benchmark zones
    plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.5)
    plt.axhline(y=20, color='red', linestyle='--', alpha=0.5)

    plt.text(0.5, 5, 'Fast Response', fontsize=10, alpha=0.7)
    plt.text(2.0, 40, 'High Throughput', fontsize=10, alpha=0.7)
    plt.text(0.5, 40, 'Optimal', fontsize=12, alpha=0.7)

    plt.tight_layout()
    pdf.savefig()
    plt.close()

def create_cost_performance_charts(df, performance_metrics, pdf):
    """Create charts showing cost vs performance tradeoffs"""
    # Reset index to make the metrics DataFrame plottable
    plot_df = performance_metrics.reset_index()

    # Check which success columns are available
    success_col = 'evaluation_success_rate'

    # 1. Create scatter plot of success rate vs cost
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        plot_df['avg_cost_per_response'],
        plot_df[success_col],
        s=200,
        c=plot_df['value_ratio'],
        cmap='viridis',
        alpha=0.7
    )

    plt.colorbar(scatter, label='Value Ratio (Success per Dollar)')

    # Add model labels to the points
    for i, row in plot_df.iterrows():
        model_name = row['model'].split('.')[-1]  # Extract the model name part
        plt.annotate(
            f"{model_name}\n({row['inference_profile']})",
            (row['avg_cost_per_response'], row[success_col]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )

    plt.title('Cost vs Performance Tradeoff', fontsize=14)
    plt.xlabel('Average Cost per Response ($)', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # 2. Create a bar chart of value ratio (performance per dollar)
    plt.figure(figsize=(14, 8))
    plot_df['model_label'] = plot_df.apply(
        lambda x: f"{x['model'].split('.')[-1]}\n({x['inference_profile']})", axis=1
    )

    ax = sns.barplot(data=plot_df, x='model_label', y='value_ratio')

    plt.title('Value Ratio (Success per Dollar)', fontsize=14)
    plt.ylabel('Success Rate per Dollar', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels on bars
    for i, v in enumerate(plot_df['value_ratio']):
        ax.text(i, v + ( v *0.05), f"{v:.0f}", ha='center', fontsize=9)

    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # 3. Cost per 1000 tokens comparison
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(data=plot_df, x='model_label', y='cost_per_1000_tokens')

    plt.title('Cost per 1000 Tokens', fontsize=14)
    plt.ylabel('Cost ($)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels on bars
    for i, v in enumerate(plot_df['cost_per_1000_tokens']):
        ax.text(i, v + ( v *0.05), f"${v:.4f}", ha='center', fontsize=9)

    plt.tight_layout()
    pdf.savefig()
    plt.close()


#----------------------------------------
# VISUALIZATION FUNCTIONS
#----------------------------------------

def create_report(output_dir, timestamp, include_llm_judge=False):
    """Generate a comprehensive PDF report from all benchmark data"""
    # Turn off interactive plotting
    plt.ioff()
    # Close any existing plots
    plt.close('all')

    pdf_file = os.path.join(output_dir, f'advanced_llm_benchmark_report_{timestamp}.pdf')

    with PdfPages(pdf_file) as pdf:
        # Create title page
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.axis('off')

        # Main title
        plt.text(0.5, 0.9, 'Advanced LLM Benchmark Report',
                 ha='center', va='center', size=24, weight='bold')

        # Timestamp
        plt.text(0.5, 0.8, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                 ha='center', va='center', size=12, style='italic', color='#666666')

        # Report sections
        sections = [
            ('Latency Metrics', 0.7, 'TTFT and OTPS measurements across models'),
            ('Performance Metrics', 0.6, 'Task success rates by model and task type'),
        ]

        if include_llm_judge:
            sections.append(('LLM Judge Evaluation', 0.5, 'Expert model assessment of responses'))

        sections.extend([
            ('Cost Analysis', 0.4, 'Cost per response and value metrics'),
            ('Integrated Analysis', 0.3, 'Combined metrics for optimal model selection')
        ])

        for section, y_pos, desc in sections:
            plt.text(0.5, y_pos, section,
                     ha='center', va='center', size=18, weight='bold', color='#2E5A88')
            plt.text(0.5, y_pos - 0.05, desc,
                     ha='center', va='center', size=12, style='italic', color='#666666')

        pdf.savefig(bbox_inches='tight', dpi=300)
        plt.close()

        print(f"Loading data from: {output_dir}")
        df = combine_csv_files(output_dir)
        df['task_success'] = df['judge_success']
        # Count error requests
        errored_requests = df[df['api_call_status'] != 'Success']
        errored_count = len(errored_requests)

        # Count throttled requests
        throttled_requests = df[df['api_call_status'] == 'ThrottlingException']
        throttled_count = len(throttled_requests)

        # Remove error requests from analysis
        df = df[df['api_call_status'] == 'Success']

        # Calculate OTPS for valid requests
        df['OTPS'] = df['model_output_tokens'] / df['time_to_last_byte']

        # Summary statistics page
        fig, ax = plt.subplots(figsize=(12, 12))
        plt.axis('off')

        # Section 1: API Statistics
        plt.text(0.1, 0.95, 'Summary Statistics', size=18, weight='bold')
        plt.text(0.1, 0.90, f"Total API calls: {len(df) + errored_count}", size=12)
        plt.text(0.1, 0.86, f"Successful calls: {len(df)}", size=12)
        plt.text(0.1, 0.82,
                 f"Errors calls: {errored_count} ({(errored_count / (len(df) + errored_count) * 100 if len(df) + errored_count > 0 else 0):.1f}%)",
                 size=12, color='#666666')
        plt.text(0.1, 0.78,
                 f"Throttled calls: {throttled_count} ({(throttled_count / (len(df) + throttled_count) * 100 if len(df) + throttled_count > 0 else 0):.1f}%)",
                 size=12, color='#666666')

        # Token Statistics section
        plt.text(0.1, 0.70, 'Token Statistics', size=18, weight='bold')
        plt.text(0.1, 0.65, f"Average Input Tokens: {df['model_input_tokens'].mean():.1f}", size=12)
        plt.text(0.1, 0.61, f"Max Input Tokens: {df['model_input_tokens'].max():.0f}", size=12)
        plt.text(0.1, 0.57, f"Average Output Tokens: {df['model_output_tokens'].mean():.1f}", size=12)
        plt.text(0.1, 0.53, f"Max Output Tokens: {df['model_output_tokens'].max():.0f}", size=12)

        # Cost Statistics
        plt.text(0.1, 0.45, 'Cost Statistics', size=18, weight='bold')
        plt.text(0.1, 0.40, f"Average Cost per Response: ${df['response_cost'].mean():.6f}", size=12)
        plt.text(0.1, 0.36, f"Total Cost of Benchmark: ${df['response_cost'].sum():.4f}", size=12)

        # Performance Statistics
        plt.text(0.1, 0.28, 'Performance Statistics', size=18, weight='bold')
        plt.text(0.1, 0.23, f"LLM Evaluation Success Rate: {df['task_success'].mean() * 100:.1f}%", size=12)

        plt.text(0.1, 0.11, f"Average Task Completion Rate: {df['success_rate'].mean() * 100:.1f}%", size=12)

        pdf.savefig(bbox_inches='tight', dpi=300)
        plt.close()

        # Calculate metrics
        performance_metrics = calculate_performance_metrics(df, ['model', 'inference_profile'])
        latency_metrics = calculate_latency_metrics(df, ['model', 'inference_profile'])

        # Create latency visualization charts
        create_latency_comparison_charts(df, latency_metrics, pdf)

        # Create performance analysis charts
        create_comparison_charts(df, pdf)

        # Create cost-performance analysis charts
        create_cost_performance_charts(df, performance_metrics, pdf)

        # Create integrated analysis charts
        create_integrated_analysis_charts(df, performance_metrics, latency_metrics, pdf)
        # Add failure analysis chart - only include failures (where task_success is False)
        failure_df = df[df['task_success'] == False]
        if len(failure_df) > 0:  # Only create chart if there are failures
            create_failure_barchart_with_indicators(failure_df, pdf)

        # Overall model recommendations page
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.axis('off')

        plt.text(0.5, 0.95, 'Model Recommendations',
                 ha='center', va='center', size=20, weight='bold')

        # Determine which success column to use
        success_col = 'evaluation_success_rate'

        try:
            # Find best models for different criteria
            best_performance = performance_metrics[success_col].idxmax()
            best_value = performance_metrics['value_ratio'].idxmax()
            best_speed = latency_metrics['TTFT_mean'].idxmin()
            best_throughput = latency_metrics['OTPS_mean'].idxmax()

            # Display recommendations
            plt.text(0.1, 0.85, 'Best for Accuracy:', size=16, weight='bold', color='#2E5A88')
            model_name, profile = best_performance
            success_rate = performance_metrics.loc[best_performance, success_col]
            plt.text(0.15, 0.80, f"{model_name} ({profile})", size=14)
            plt.text(0.15, 0.75, f"Success Rate: {success_rate * 100:.1f}%", size=12, color='#666666')

            plt.text(0.1, 0.65, 'Best Value (Performance per Dollar):', size=16, weight='bold', color='#2E5A88')
            model_name, profile = best_value
            value_ratio = performance_metrics.loc[best_value, 'value_ratio']
            success_rate = performance_metrics.loc[best_value, success_col]
            cost = performance_metrics.loc[best_value, 'avg_cost_per_response']
            plt.text(0.15, 0.60, f"{model_name} ({profile})", size=14)
            plt.text(0.15, 0.55, f"Success Rate: {success_rate * 100:.1f}%, Cost: ${cost:.6f}", size=12,
                     color='#666666')

            plt.text(0.1, 0.45, 'Best for Speed (TTFT):', size=16, weight='bold', color='#2E5A88')
            model_name, profile = best_speed
            ttft = latency_metrics.loc[best_speed, 'TTFT_mean']
            plt.text(0.15, 0.40, f"{model_name} ({profile})", size=14)
            plt.text(0.15, 0.35, f"TTFT: {ttft:.3f} seconds", size=12, color='#666666')

            plt.text(0.1, 0.25, 'Best for Throughput (OTPS):', size=16, weight='bold', color='#2E5A88')
            model_name, profile = best_throughput
            otps = latency_metrics.loc[best_throughput, 'OTPS_mean']
            plt.text(0.15, 0.20, f"{model_name} ({profile})", size=14)
            plt.text(0.15, 0.15, f"OTPS: {otps:.2f} tokens/second", size=12, color='#666666')
        except Exception as e:
            plt.text(0.5, 0.5, f"Insufficient data to generate recommendations: {str(e)}",
                     ha='center', va='center', size=14, color='red')

        pdf.savefig(bbox_inches='tight', dpi=300)
        plt.close()

        # Save metrics to CSV
        perf_csv_file = os.path.join(output_dir, f'performance_metrics_{timestamp}.csv')
        performance_metrics.to_csv(perf_csv_file)

        latency_csv_file = os.path.join(output_dir, f'latency_metrics_{timestamp}.csv')
        latency_metrics.to_csv(latency_csv_file)

        print(f"\nReport generation complete!")
        print(f"PDF report saved to: {pdf_file}")
        print(f"Performance metrics saved to: {perf_csv_file}")
        print(f"Latency metrics saved to: {latency_csv_file}")

    return pdf_file


def combine_csv_files(directory):
    """Combine all CSV files in the directory into a single DataFrame."""
    all_files = glob.glob(os.path.join(directory, "invocations_*.csv"))
    df_list = []
    for filename in all_files:
        df = pd.read_csv(filename)
        df_list.append(df)
    return pd.concat(df_list, axis=0, ignore_index=True)


# ----------------------------------------
# METRICS CALCULATION AND VISUALIZATION
# ----------------------------------------

def calculate_performance_metrics(df, group_columns):
    """Calculate comprehensive performance metrics grouped by specified columns."""
    # Prepare metrics to calculate
    agg_dict = {
        'task_success': ['count', 'mean'],
        'success_rate': ['mean'],
        'response_cost': ['mean', 'sum'],
        'model_output_tokens': ['mean'],
        'model_input_tokens': ['mean']
    }

    # Calculate metrics
    metrics = df.groupby(group_columns).agg(agg_dict).round(4)

    # Rename columns for clarity
    column_names = ['sample_size', 'evaluation_success_rate', 'avg_task_success_rate',
                    'avg_cost_per_response', 'total_cost',
                    'avg_output_tokens', 'avg_input_tokens']

    metrics.columns = column_names

    # Calculate value metrics (success per dollar)
    success_col = 'evaluation_success_rate'
    metrics['value_ratio'] = metrics[success_col] / metrics['avg_cost_per_response']
    metrics['cost_per_1000_tokens'] = (metrics['avg_cost_per_response'] /
                                       (metrics['avg_output_tokens'] + metrics['avg_input_tokens'])) * 1000

    return metrics


def calculate_latency_metrics(df, group_columns):
    """Calculate latency metrics from the unified benchmark dataset."""
    metrics = df.groupby(group_columns).agg({
        'time_to_first_byte': ['count', 'mean', 'median',
                               lambda x: x.quantile(0.9),
                               lambda x: x.std()],
        'time_to_last_byte': ['mean', 'median',
                              lambda x: x.quantile(0.9)]
    }).round(3)

    metrics.columns = ['sample_size', 'TTFT_mean', 'TTFT_p50', 'TTFT_p90', 'TTFT_std',
                       'total_time_mean', 'total_time_p50', 'total_time_p90']

    # Calculate OTPS (output tokens per second)
    df['OTPS'] = df['model_output_tokens'] / df['time_to_last_byte']
    otps_metrics = df.groupby(group_columns)['OTPS'].agg(['mean', 'median',
                                                          lambda x: x.quantile(0.9),
                                                          lambda x: x.std()]).round(3)
    otps_metrics.columns = ['OTPS_mean', 'OTPS_p50', 'OTPS_p90', 'OTPS_std']

    metrics = pd.concat([metrics, otps_metrics], axis=1)
    return metrics


def create_failure_barchart_with_indicators(failure_df, pdf):
    """
    Create bar charts showing failures by model for each task type with indicators for failure reasons.
    Each task type gets its own chart.

    Parameters:
    -----------
    failure_df : DataFrame
        DataFrame containing failure data with columns: model, task_types, judge_explanation
    pdf : PdfPages
        PDF document to add the charts to
    """
    # Split combined judgment codes into individual failures
    expanded_failures = []

    for _, row in failure_df.iterrows():
        # Split combined error codes
        if isinstance(row['judge_explanation'], str) and ',' in row['judge_explanation']:
            error_types = row['judge_explanation'].split(',')
            for error in error_types:
                new_row = row.copy()
                new_row['judge_explanation'] = error
                expanded_failures.append(new_row)
        else:
            expanded_failures.append(row)

    expanded_df = pd.DataFrame(expanded_failures)

    # Process each task type separately
    for task_type in expanded_df['task_types'].unique():
        # Filter for current task type
        task_df = expanded_df[expanded_df['task_types'] == task_type]

        # Get unique models and error types for this task
        models = task_df['model'].unique()
        error_types = task_df['judge_explanation'].unique()

        # Create a dynamic colormap based on number of error types
        cmap = plt.cm.get_cmap('tab20', len(error_types))
        error_colors = {error_type: cmap(i) for i, error_type in enumerate(error_types)}

        # Count failures by model and error type
        failure_counts = task_df.groupby(['model', 'judge_explanation']).size().reset_index(name='count')

        # Prepare plot data
        plot_data = pd.DataFrame(index=models)

        # Count total failures by model
        total_failures = failure_counts.groupby('model')['count'].sum()
        plot_data['total'] = total_failures

        # Create the figure
        fig, ax = plt.subplots(figsize=(14, 8))

        # Calculate bar positions
        x = np.arange(len(models))

        # Create model labels
        model_labels = [m.split('.')[-1] for m in models]

        # Plot stacked bars for each error type
        bottom = np.zeros(len(models))

        # Dictionary to hold the bars for legend
        bars_for_legend = {}

        # Sort error types by frequency for better visualization
        error_counts = task_df['judge_explanation'].value_counts()
        sorted_errors = error_counts.index.tolist()

        for error in sorted_errors:
            # Get counts for this error type by model
            error_data = failure_counts[failure_counts['judge_explanation'] == error]

            # Create array of counts matching the order of models
            counts = np.zeros(len(models))
            for i, model in enumerate(models):
                model_row = error_data[error_data['model'] == model]
                if not model_row.empty:
                    counts[i] = model_row['count'].values[0]

            # Create the stacked bar
            bar = ax.bar(x, counts, bottom=bottom,
                         label=error, color=error_colors[error])

            # Keep track for legend
            bars_for_legend[error] = bar

            # Update bottom for next stack
            bottom += counts

        # Add labels, title and legend
        ax.set_xlabel('Model')
        ax.set_ylabel('Failure Count')
        ax.set_title(f'Failure Analysis for {task_type} Tasks')
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=45, ha='right')

        # Add legend
        ax.legend(title="Failure Types")

        # Add total count labels on top of bars
        for i, total in enumerate(total_failures):
            if total > 0:
                ax.text(i, total + 0.1, f'{int(total)}', ha='center')

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()