import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

def plot_forecast_results(forecast_results_by_state, state, title_prefix="Dengue Forecast"):
    """
    Plot probabilistic dengue forecasts with confidence intervals.
    
    Parameters:
    -----------
    forecast_results_by_state : dict
        Dictionary containing forecast results for each state
        Structure: {state: {'forecast': DataFrame, 'observed': DataFrame}}
    state : str
        State name to plot
    title_prefix : str
        Prefix for plot title
    """
    
    # Extract forecast and observed data for the specified state
    forecast_df = forecast_results_by_state[state]['forecast']
    observed_df = forecast_results_by_state[state]['observed']
    
    # Convert dates to datetime if needed
    forecast_df['date'] = pd.to_datetime(forecast_df['date'])
    observed_df['data_iniSE'] = pd.to_datetime(observed_df['data_iniSE'])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 1. Plot confidence intervals (from widest to narrowest for visual layering)
    confidence_levels = [95, 90, 80, 50]
    colors = {
        95: '#9ecae1',  # Lightest blue
        90: '#6baed6',
        80: '#4292c6',
        50: '#2171b5'   # Darkest blue
    }
    
    # Plot confidence intervals as shaded regions
    for level in confidence_levels:
        lower_col = f'lower_{level}'
        upper_col = f'upper_{level}'
        
        ax.fill_between(
            forecast_df['date'],
            forecast_df[lower_col],
            forecast_df[upper_col],
            color=colors[level],
            alpha=0.25 if level == 95 else 0.35 if level == 90 else 0.45 if level == 80 else 0.55,
            label=f'{level}% PI' if level in [95, 50] else None,  # Only label widest and narrowest
            zorder=1
        )
    
    # 2. Plot median forecast
    ax.plot(
        forecast_df['date'],
        forecast_df['pred'],
        color='#3182bd',
        linewidth=2.5,
        label='Forecast Median',
        zorder=2
    )
    
    # 3. Plot observed data
    # Split observed data into training and forecast periods
    # Assuming the forecast starts after the last observed data point
    # Find the split point (forecast start date)
    forecast_start = forecast_df['date'].min()
    
    # Training data (before forecast start)
    train_mask = observed_df['data_iniSE'] < forecast_start
    # Forecast period data (during or after forecast start)
    forecast_mask = observed_df['data_iniSE'] >= forecast_start
    
    # Plot training data (filled circles)
    if train_mask.any():
        ax.scatter(
            observed_df.loc[train_mask, 'data_iniSE'],
            observed_df.loc[train_mask, 'casos'],
            color='black',
            s=50,
            label='Observed (Training)',
            zorder=3
        )
    
    # Plot forecast period observed data (hollow circles)
    if forecast_mask.any():
        ax.scatter(
            observed_df.loc[forecast_mask, 'data_iniSE'],
            observed_df.loc[forecast_mask, 'casos'],
            facecolors='none',
            edgecolors='black',
            s=50,
            linewidths=1.5,
            label='Observed (Forecast Period)',
            zorder=3
        )
    
    # 4. Add vertical line to separate training and forecast periods
    if train_mask.any() and forecast_mask.any():
        ax.axvline(
            x=forecast_start,
            color='red',
            linestyle='--',
            linewidth=1.5,
            alpha=0.6,
            label='Forecast Start'
        )
    
    # 5. Customize the plot
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Detected Cases', fontsize=12)
    ax.set_title(f'{title_prefix} - {state}', fontsize=14, fontweight='bold')
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 6. Create custom legend
    # Create custom legend elements
    legend_elements = [
        # Observed data (training)
        Line2D([0], [0], marker='o', linestyle='None', 
               markerfacecolor='black', markeredgecolor='black', markersize=8,
               label='Observed (Training)'),
        # Observed data (forecast period)
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor='none', markeredgecolor='black', markersize=8,
               label='Observed (Forecast Period)'),
        # Forecast median
        Line2D([0], [0], color='#3182bd', lw=2.5, label='Forecast Median'),
        # Confidence intervals
        Line2D([0], [0], color='#9ecae1', lw=10, alpha=0.35, label='95% PI'),
        Line2D([0], [0], color='#2171b5', lw=10, alpha=0.55, label='50% PI'),
        # Forecast start line
        Line2D([0], [0], color='red', linestyle='--', lw=1.5, alpha=0.6, label='Forecast Start')
    ]
    
    # Add legend with custom handling
    ax.legend(
        handles=legend_elements,
        loc='upper left',
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95
    )
    
    # 7. Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)  # Put grid behind data
    
    # 8. Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    # Return the figure for additional customization if needed
    return fig, ax


def plot_multiple_states_forecast(forecast_results_by_state, states=None, 
                                  nrows=2, ncols=2, title_prefix="Dengue Forecast"):
    """
    Plot forecasts for multiple states in a grid layout.
    
    Parameters:
    -----------
    forecast_results_by_state : dict
        Dictionary containing forecast results for each state
    states : list, optional
        List of states to plot. If None, plots all states.
    nrows, ncols : int
        Number of rows and columns for subplot grid
    title_prefix : str
        Prefix for plot titles
    """
    
    if states is None:
        states = list(forecast_results_by_state.keys())
    
    n_states = len(states)
    n_plots = min(n_states, nrows * ncols)
    
    # Create subplot grid
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    confidence_levels = [95, 90, 80, 50]
    colors = {
        95: '#9ecae1',
        90: '#6baed6',
        80: '#4292c6',
        50: '#2171b5'
    }
    
    for idx, state in enumerate(states[:n_plots]):
        ax = axes[idx]
        
        # Extract data
        forecast_df = forecast_results_by_state[state]['forecast']
        observed_df = forecast_results_by_state[state]['observed']
        
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        observed_df['data_iniSE'] = pd.to_datetime(observed_df['data_iniSE'])
        
        # Plot confidence intervals
        for level in confidence_levels:
            lower_col = f'lower_{level}'
            upper_col = f'upper_{level}'
            
            ax.fill_between(
                forecast_df['date'],
                forecast_df[lower_col],
                forecast_df[upper_col],
                color=colors[level],
                alpha=0.25 if level == 95 else 0.35 if level == 90 else 0.45 if level == 80 else 0.55,
                zorder=1
            )
        
        # Plot median forecast
        ax.plot(
            forecast_df['date'],
            forecast_df['pred'],
            color='#3182bd',
            linewidth=2,
            label='Forecast',
            zorder=2
        )
        
        # Plot observed data
        forecast_start = forecast_df['date'].min()
        train_mask = observed_df['data_iniSE'] < forecast_start
        forecast_mask = observed_df['data_iniSE'] >= forecast_start
        
        if train_mask.any():
            ax.scatter(
                observed_df.loc[train_mask, 'data_iniSE'],
                observed_df.loc[train_mask, 'casos'],
                color='black',
                s=30,
                zorder=3
            )
        
        if forecast_mask.any():
            ax.scatter(
                observed_df.loc[forecast_mask, 'data_iniSE'],
                observed_df.loc[forecast_mask, 'casos'],
                facecolors='none',
                edgecolors='black',
                s=30,
                linewidths=1.5,
                zorder=3
            )
        
        # Add vertical line
        if train_mask.any() and forecast_mask.any():
            ax.axvline(
                x=forecast_start,
                color='red',
                linestyle='--',
                linewidth=1,
                alpha=0.5
            )
        
        # Customize subplot
        ax.set_title(state, fontsize=11, fontweight='bold')
        ax.set_xlabel('Date', fontsize=9)
        ax.set_ylabel('Cases', fontsize=9)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    # Add overall title
    fig.suptitle(f'{title_prefix} - Multiple States', fontsize=14, fontweight='bold', y=1.02)
    
    # Create legend for the entire figure
    legend_elements = [
        Line2D([0], [0], color='#3182bd', lw=2, label='Forecast Median'),
        Line2D([0], [0], color='#9ecae1', lw=8, alpha=0.35, label='95% PI'),
        Line2D([0], [0], color='#2171b5', lw=8, alpha=0.55, label='50% PI'),
        Line2D([0], [0], marker='o', linestyle='None', 
               markerfacecolor='black', markeredgecolor='black', markersize=6,
               label='Observed (Training)'),
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor='none', markeredgecolor='black', markersize=6,
               label='Observed (Forecast)'),
        Line2D([0], [0], color='red', linestyle='--', lw=1, alpha=0.5, label='Forecast Start')
    ]
    
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=3,
        bbox_to_anchor=(0.5, -0.05),
        frameon=True,
        fancybox=True,
        shadow=True
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()
    
    return fig, axes
