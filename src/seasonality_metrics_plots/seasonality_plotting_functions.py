# %%
import matplotlib.pyplot as plt
import seaborn as sns

import math
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

# for mapped thresholds
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.cm as cm


# %%
# helper functions
def safely_add_yearmon_column_to_df(df):
    if 'year' not in df.columns or 'month' not in df.columns:
        raise ValueError("DataFrame must contain 'year' and 'month' columns.")
    
    if 'yearmon' in df.columns:
        print(f"'yearmon' column already exists in supplied DataFrame")    
    else:        
        df['yearmon'] = df['year'] + ((df['month'] - 1) / 12)
        df['yearmon'] = df['yearmon'].astype(np.float32)
    
    return df



# %%
def get_nice_breaks_from_list_of_df_columns(
    df, 
    column_names: list[str], 
    min_value=None, 
    min_breaks=3, 
    max_breaks=5, 
    buffer=1.1
):
    # Compute min and max across all specified columns, ignoring NaNs
    raw_min = df[column_names].min().min()
    raw_max = df[column_names].max().max()

    # Use computed min if none was specified
    if min_value is None:
        min_value = raw_min

    max_value = raw_max * buffer if raw_max > 0 else raw_max

    if not np.isfinite(max_value) or not np.isfinite(min_value) or max_value <= min_value:
        return np.array([min_value, min_value + 1])  # fallback option

    # Preferred "nice" step sizes
    base_steps = np.array([1, 2, 5])
    exponents = np.arange(-2, 5)  # e.g., 0.01 to 10000
    all_steps = np.sort(np.outer(base_steps, 10.0**exponents).flatten())

    for step in all_steps:
        if step <= 0:
            continue
        try:
            lower_bound = np.floor(min_value / step) * step
            upper_bound = np.ceil(max_value / step) * step
            breaks = np.arange(lower_bound, upper_bound + 0.5 * step, step)
            num_breaks = len(breaks)

            if min_breaks <= num_breaks <= max_breaks:
                # Determine appropriate decimal precision
                range_span = max_value - min_value
                if range_span >= 1:
                    decimals = 0
                elif range_span >= 0.1:
                    decimals = 1
                else:
                    decimals = 2

                breaks = np.around(breaks, decimals=decimals)
                breaks = np.unique(breaks)  # Remove duplicates after rounding

                if np.allclose(breaks, breaks.astype(int)):
                    breaks = breaks.astype(int)

                return breaks
        except Exception:
            continue  # skip problematic step sizes

    # Fallback option: linspace with smart rounding
    breaks = np.linspace(min_value, max_value, max_breaks)

    range_span = max_value - min_value
    if range_span >= 1:
        decimals = 0
    elif range_span >= 0.1:
        decimals = 1
    else:
        decimals = 2

    breaks = np.around(breaks, decimals=decimals)
    breaks = np.unique(breaks)

    if np.allclose(breaks, breaks.astype(int)):
        breaks = breaks.astype(int)

    return breaks


# %%

def plot_monthly_data_with_optional_year_overlay(
        plot_df,
        outcome_column,
        time_axis_column = 'yearmon',
        plot_monthly_data_overlay_years = False,
        use_supplied_outcome_breaks = False,
        supplied_outcome_breaks = None
    ):

    fig, ax = plt.subplots(figsize=(15, 5)) 
    unique_years = plot_df['year'].unique()

    if plot_monthly_data_overlay_years == False:
        sns.lineplot(
            data=plot_df,
            x=time_axis_column,
            y=outcome_column,
            alpha=0.7,
            errorbar=None,
            ax=ax,
            legend=False
        )    
        ax.set_xlabel("\nYear and Month", fontsize=14)
        year_start_mask = plot_df['month'] == 1
        xticks = plot_df.loc[year_start_mask, 'yearmon'].drop_duplicates()
        xtick_labels = plot_df.loc[year_start_mask, 'year'].drop_duplicates().astype(str)

    if plot_monthly_data_overlay_years == True:
        overlay_palette = sns.color_palette('husl', len(unique_years))
        sns.lineplot(
            data=plot_df, 
            x='month', 
            y= outcome_column, 
            hue='year',
            palette = overlay_palette,
            linewidth=2,
            errorbar=None  
        )
        plt.legend(title='Year\n', labels=[str(year) for year in unique_years])
        ax.set_xlabel("\nMonth", fontsize=14)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), title="Year\n")
        xticks = plot_df['month'].drop_duplicates().tolist()
        xtick_labels = plot_df['month'].drop_duplicates().tolist()

    # set the ticks for either of the two plot options
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

    if use_supplied_outcome_breaks:
        ax.set_yticks(supplied_outcome_breaks)    
        ax.set_yticklabels(supplied_outcome_breaks)
    else: 
        plot_df_outcome_breaks = get_nice_breaks_from_list_of_df_columns(
            plot_df, 
            [outcome_column], 
            min_value = None, 
            min_breaks=3, 
            max_breaks=5,
        )
        ax.set_yticks(plot_df_outcome_breaks)    
        ax.set_yticklabels(plot_df_outcome_breaks)

    # suppress y label and add manually after call
    ax.set_ylabel("")

    return fig, ax

### Example use: 
# fig, ax = plot_monthly_data_with_optional_overlay(
#     plot_df=filtered_df,
#     outcome_column=outcome_column,
#     time_axis_column='yearmon',
#     plot_monthly_data_overlay_years= plot_monthly_data_overlay_years,
#     use_supplied_outcome_breaks = use_same_y_axis_for_all_areas_in_country,
#     supplied_outcome_breaks = df_outcome_breaks
# )

### Then either of
# st.pyplot(fig, width='content')
# plt.show()

# %% --------------------------------------------------------------------------
# plot monthly data with optional year overlay (or second outcome series)

def plot_monthly_data_with_optional_year_overlay_or_second_outcome(
        plot_df,
        outcome_column,
        time_axis_column = 'yearmon',
        plot_monthly_data_overlay_years = False,
        use_supplied_outcome_breaks = False,
        supplied_outcome_breaks = None,
        second_outcome = None,
        use_supplied_second_outcome_breaks = False,
        supplied_second_outcome_breaks = None
    ):

    fig, ax = plt.subplots(figsize=(15, 5)) 
    unique_years = plot_df['year'].unique()

    palette = sns.color_palette("muted")
    outcome_colour = palette[0]    
    second_outcome_colour = palette[1] 

    if plot_monthly_data_overlay_years == False:
        sns.lineplot(
            data=plot_df,
            x=time_axis_column,
            y=outcome_column,
            color = outcome_colour,
            errorbar=None,
            ax=ax,
            legend=False
        )    
        ax.set_xlabel("\nYear and Month", fontsize=14)
        year_start_mask = plot_df['month'] == 1
        xticks = plot_df.loc[year_start_mask, 'yearmon'].drop_duplicates()
        xtick_labels = plot_df.loc[year_start_mask, 'year'].drop_duplicates().astype(str)

        if second_outcome:
            # Create secondary y-axis
            ax2 = ax.twinx()

            # Plot the second series on the right axis
            sns.lineplot(
                data=plot_df,
                x=time_axis_column,
                y=second_outcome,  # replace with actual column name
                errorbar=None,
                color= second_outcome_colour,
                ax=ax2,
                legend=False
            )

            # Label second axis
            ax2.set_ylabel("")

    if plot_monthly_data_overlay_years == True:
        overlay_palette = sns.color_palette('husl', len(unique_years))
        sns.lineplot(
            data=plot_df, 
            x='month', 
            y= outcome_column, 
            hue='year',
            palette = overlay_palette,
            linewidth=2,
            errorbar=None  
        )
        plt.legend(title='Year\n', labels=[str(year) for year in unique_years])
        ax.set_xlabel("\nMonth", fontsize=14)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), title="Year\n")
        xticks = plot_df['month'].drop_duplicates().tolist()
        xtick_labels = plot_df['month'].drop_duplicates().tolist()

    # set the ticks for either of the two plot options
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

    if use_supplied_outcome_breaks:
        ax.set_yticks(supplied_outcome_breaks)    
        ax.set_yticklabels(supplied_outcome_breaks)
    else: 
        plot_df_outcome_breaks = get_nice_breaks_from_list_of_df_columns(
            plot_df, 
            [outcome_column], 
            min_value = None, 
            min_breaks=3, 
            max_breaks=5,
        )
        ax.set_yticks(plot_df_outcome_breaks)    
        ax.set_yticklabels(plot_df_outcome_breaks)

    # For the secondary axis (ax2)
    if second_outcome:
        if use_supplied_second_outcome_breaks:
            ax2.set_yticks(supplied_second_outcome_breaks)
            ax2.set_yticklabels(supplied_second_outcome_breaks)
        else:
            plot_df_secondary_breaks = get_nice_breaks_from_list_of_df_columns(
                plot_df, 
                [second_outcome],  # replace with your actual column
                min_value=None, 
                min_breaks=3, 
                max_breaks=5,
            )
            ax2.set_yticks(plot_df_secondary_breaks)
            ax2.set_yticklabels(plot_df_secondary_breaks)

    # suppress y label and add manually after call
    ax.set_ylabel("")

    # improve appearance when there are many ticks
    if len(xticks)>20:
        ax.tick_params(axis='x', labelrotation=90)

    axes_flat = [ax]
    
    if second_outcome and plot_monthly_data_overlay_years != True:
        axes_flat.append(ax2)
    return fig, axes_flat



# %% --------------------------------------------------------------------------
# plot percentage of annual total by month

def plot_percentage_of_annual_totals_by_month_or_in_windows(
    seasonality_metrics_df,
    overlay_years = False,
    plot_percentages_in_window = False,
    window_size = 4    
    ):

    fig, ax = plt.subplots(figsize=(15, 5))

    unique_years = seasonality_metrics_df['year'].unique()
    years_palette = sns.color_palette('husl', len(unique_years))

    if plot_percentages_in_window == False:
        if overlay_years == False:
            sns.lineplot(
                data=seasonality_metrics_df,
                x='yearmon',
                y='month_percent',
                alpha=0.7,
                errorbar=None,
                ax=ax,
                legend=False
            )    
            ax.set_title(f"Percent of Annual Total by Month - Full Time Series\n", fontsize=14)
            ax.set_xlabel("\nYear and Month", fontsize=14)
            ax.set_ylabel("Percent of Annual Total in Each Month\n", fontsize=14)
            year_start_mask = seasonality_metrics_df['month'] == 1
            xticks = seasonality_metrics_df.loc[year_start_mask, 'yearmon'].drop_duplicates()
            xtick_labels = seasonality_metrics_df.loc[year_start_mask, 'year'].drop_duplicates().astype(str)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels)

        if overlay_years == True:
            sns.lineplot(
                data=seasonality_metrics_df, 
                x='month', 
                y= 'month_percent', 
                hue='year',
                palette = years_palette,
                linewidth=2,
                errorbar=None  
            )
            plt.legend(title='Year\n', labels=[str(year) for year in unique_years])
            ax.set_title(f"Percent of Annual Total by Month - Overlaying Separate Years\n", fontsize=14)
            ax.set_xlabel("\nMonth", fontsize=14)
            ax.set_ylabel("Percent of Annual Total in Each Month\n", fontsize=14)
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), title="Year\n")
            xticks = seasonality_metrics_df['month'].drop_duplicates().tolist()
            xtick_labels = seasonality_metrics_df['month'].drop_duplicates().tolist()
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels)

    # NB - assume if window percentages are plotted, years are overlayed
    # (so overlay_years is ignored)
    if plot_percentages_in_window == True:
        
        window_size_mask = seasonality_metrics_df['window_size'] == window_size 
        seasonality_metrics_df_for_specified_window_size = seasonality_metrics_df.loc[window_size_mask].copy()   

        sns.lineplot(
            data=seasonality_metrics_df_for_specified_window_size, 
            x='month', 
            y= 'window_percentage', 
            hue='year',
            palette = years_palette,
            linewidth=2,
            errorbar=None  
        )
        plt.legend(title='Year\n', labels=[str(year) for year in unique_years])
        ax.set_title(f"Percentage of Annual Total in {window_size} Month Window by Start Month\n", fontsize=14)
        ax.set_xlabel(f"\nStart Month - {window_size} Month Window", fontsize=14)
        ax.set_ylabel("Percent of Annual Total in Window\n", fontsize=14)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), title="Year\n")
        xticks = seasonality_metrics_df['month'].drop_duplicates().tolist()
        xtick_labels = seasonality_metrics_df['month'].drop_duplicates().tolist()
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels) 

    return fig, ax



# %% --------------------------------------------------------------------------
# Start month consistency histograms

def plot_start_month_consistency_histogram_using_year_summary_df(
    year_summary_df,
    window_size
    ):

    # filter year summary df to specified window size
    window_size_mask = year_summary_df['window_size'] == window_size 
    year_summary_df_for_specified_window_size = year_summary_df.loc[window_size_mask].copy()   

    histo_palette = sns.color_palette()
    sns_blue = histo_palette[0]

    fig, ax = plt.subplots(figsize=(15, 5))
    
    sns.histplot(
        data=year_summary_df_for_specified_window_size,
        x='max_start_month',
        bins=np.arange(0.5, 13.5, 1),  # to center bars on integers 1–12
        discrete=True,
        shrink=0.8,
        color=sns_blue
    )

    # Customize axes
    ax.set_xticks(range(1, 13))
    ax.set_xlim(0.5, 12.5)
    ax.set_xlabel(f"\nStart Month of {window_size} Month Window", fontsize=14)
    ax.set_ylabel("Number of Years\nPeak Window Starts in Each Month\n", fontsize=14)
    ax.set_title(f"Consistency of Start Month for Peak {window_size} Month Window\n", fontsize=14)

    return fig, ax



# %% --------------------------------------------------------------------------
# polar seasonality plot - percentage by month, with polar co-ordinates

def make_polar_seasonality_plot(
        seasonality_metrics_df,
        polar_title_note = None,
    ):

    polar_df = seasonality_metrics_df.copy()
    polar_df["month"] = polar_df["month"].astype(int)

    # Compute theta: 12 = 0°, 1 = 30°, ..., 11 = 60°
    polar_df["theta"] = ((12 - polar_df["month"]) % 12) * (2 * np.pi / 12)

    # Set up plot
    fig, ax = plt.subplots(figsize=(15, 15), subplot_kw={'polar': True})
    unique_years = sorted(polar_df['year'].unique())
    palette = sns.color_palette('husl', len(unique_years))

    # Plot each year, appending first point at the end to close the loop
    for i, year in enumerate(unique_years):
        data = polar_df[polar_df["year"] == year].copy()
        data = data.sort_values("month")

        # Append the first row to the end (phantom January)
        first_row = data.iloc[0:1].copy()
        data = pd.concat([data, first_row])

        ax.plot(
            data["theta"], 
            data["month_percent"], 
            label=str(year), 
            linewidth=2, 
            color=palette[i]
        )

    # Polar axis settings
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(1)

    # Month labels (1–12)
    month_angles = ((12 - np.arange(1, 13)) % 12) * (2 * np.pi / 12)
    month_labels = [str(m) for m in range(1, 13)]
    ax.set_xticks(month_angles)
    ax.set_xticklabels(month_labels, fontsize=12)

    # Radial limits
    ax.set_ylim(0, polar_df['month_percent'].max() * 1.1)
    ax.set_rlabel_position(0)
    ax.set_ylabel("Monthly % of Annual\n\n", fontsize=16)

    # Legend and title
    ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.05), title="Year", fontsize=16, title_fontsize=16)
    plt.title(f"{'\n'*10}Polar Seasonality Plot for {polar_title_note}\n", fontsize=20, pad=20)

    return fig, ax


# %% --------------------------------------------------------------------------
# markham seaonality index polygons
def plot_markham_seasonality_index_polygons(df_filtered, area_column_list, facet_years=False):
    
    # create angle_table_df
    angle_table_df = pd.DataFrame({
    'month': np.arange(1, 13),
    'midpoint_as_degrees': [
        15.287671, 44.383560, 73.479454, 103.561646, 133.643829, 163.726028,
        193.808212, 224.383560, 254.465759, 284.547943, 314.630127, 344.712341
        ]
    })
    angle_table_df['gradient'] = np.tan(np.deg2rad(angle_table_df['midpoint_as_degrees'] - 90))
    
    # Merge in angles
    df_filtered = df_filtered.merge(angle_table_df, on='month', how='left')

    unique_years = sorted(df_filtered['year'].unique())
    num_years = len(unique_years)
    polygon_palette = sns.color_palette('husl', num_years)

    if facet_years:
        # Determine number of rows and columns for subplot grid
        n_cols = math.ceil(math.sqrt(num_years))
        n_rows = math.ceil(num_years / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15), squeeze=False)
    else:
        fig, ax = plt.subplots(figsize=(15, 15))

    for i, year in enumerate(unique_years):
        plot_df = df_filtered[df_filtered['year'] == year].copy()
        needed_cols = [*area_column_list, 'month', 'month_proportion', 'midpoint_as_degrees', 'gradient']

        origin_row = plot_df.iloc[[0]].copy()
        origin_row[['month', 'month_proportion', 'midpoint_as_degrees', 'gradient']] = 0
        polygon_df = pd.concat([origin_row[needed_cols], plot_df[needed_cols]], ignore_index=True)

        angle_radians = np.deg2rad(90 - polygon_df['midpoint_as_degrees'])
        polygon_df['dx'] = polygon_df['month_proportion'] * np.cos(angle_radians)
        polygon_df['dy'] = polygon_df['month_proportion'] * np.sin(angle_radians)

        polygon_df['x'] = polygon_df['dx'].cumsum()
        polygon_df['y'] = polygon_df['dy'].cumsum()
        polygon_df['x0'] = polygon_df['x'].shift(fill_value=0)
        polygon_df['y0'] = polygon_df['y'].shift(fill_value=0)

        if facet_years:
            row_idx, col_idx = divmod(i, n_cols)
            ax = axes[row_idx][col_idx]
            alpha_months = 0.7
            alpha_resultant = 1.0
        else:
            ax = ax  # reuse single axis
            alpha_months = 0.5
            alpha_resultant = 0.7

        # Draw arrows
        for _, row in polygon_df[polygon_df['month'] != 0].iterrows():
            ax.arrow(row['x0'], row['y0'], row['dx'], row['dy'],
                     head_width=0.01, 
                     head_length=0.01, 
                     fc=polygon_palette[i], 
                     ec=polygon_palette[i],
                     linestyle='dashed', 
                     alpha=alpha_months)

        # Resultant vector
        xfinal = polygon_df['x'].iloc[-1]
        yfinal = polygon_df['y'].iloc[-1]
        ax.arrow(0, 0, xfinal, yfinal,
                 head_width=0.02, 
                 head_length=0.02,
                 label=str(year),
                 fc=polygon_palette[i], 
                 ec=polygon_palette[i], 
                 alpha=alpha_resultant, 
                 lw=1)

        ax.set_aspect('equal')
        ax.axhline(0, color=(0.5, 0.5, 0.5, 0.05), lw=0.5)
        ax.axvline(0, color=(0.5, 0.5, 0.5, 0.05), lw=0.5)
        ax.set_title(f'{year}')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.grid(True)

    if facet_years:
        # Hide any unused subplots
        for j in range(i + 1, n_rows * n_cols):
            row_idx, col_idx = divmod(j, n_cols)
            fig.delaxes(axes[row_idx][col_idx])
        fig.suptitle(f'Markham Polygons by Year\n', fontsize=16)
        fig.tight_layout()
        fig.subplots_adjust(top=0.92)
        axes_flat = axes.flatten()
    
    else:
        ax.set_title(f'Markham Seasonality Polygons Overlaying Years\n', fontsize = 16)
        ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.05), title="Year\n", fontsize=16, title_fontsize=16)
        axes_flat = [ax]

    # return axes as a flat list for either overlay or facet
    return fig, axes_flat


# %% --------------------------------------------------------------------------
# plot STL decomposition with optional fitted line

def plot_STL_decomposition_with_optional_seasonal_plus_mean_trend(
    df_for_stl_plot,
    outcome_column,
    add_seasonal_plus_mean_trend_to_plot = True,
    display_trend_on_same_scale_as_data = True,
    stl_title_note = None,
    stl_period = 12,
    return_stl_results_df = False     
    ):

    df_for_stl_plot = df_for_stl_plot.copy()

    if stl_title_note is None:
        stl_title_note = f"{outcome_column}"

    seasonal_smoother_length = stl_period * 2 + 1

    # Perform STL decomposition
    stl = STL(df_for_stl_plot[outcome_column].values, seasonal = seasonal_smoother_length, period=stl_period, robust=True)
    stl_result = stl.fit()

    # Extract components
    df_for_stl_plot['trend'] = stl_result.trend
    df_for_stl_plot['seasonal'] = stl_result.seasonal
    df_for_stl_plot['resid'] = stl_result.resid

    # Create processed column: seasonal + mean of trend
    mean_trend = df_for_stl_plot['trend'].mean()
    df_for_stl_plot['seasonal_plus_mean_trend'] = df_for_stl_plot['seasonal'] + mean_trend

    # STL decomposition plot
    fig, ax = plt.subplots(4, 1, figsize=(15, 15), sharex=True)

    stl_palette = sns.color_palette()

    plot_min = df_for_stl_plot[outcome_column].min()
    plot_max = df_for_stl_plot[outcome_column].max()

    # Observed
    sns.lineplot(
        data=df_for_stl_plot,
        x='yearmon',
        y=outcome_column,
        color = stl_palette[0],
        errorbar=None,
        ax=ax[0],
        legend=False
    )
    ax[0].set_ylabel("Data", fontsize=12)

    # Trend
    sns.lineplot(
        data=df_for_stl_plot,
        x='yearmon',
        y='trend',
        ax=ax[1],
        color=stl_palette[1],
        legend=False
    )
    ax[1].set_ylabel("Trend", fontsize=12)
    if display_trend_on_same_scale_as_data:
        ax[1].set_ylim(plot_min, plot_max)

    # Seasonal
    sns.lineplot(
        data=df_for_stl_plot,
        x='yearmon',
        y='seasonal',
        ax=ax[2],
        color=stl_palette[2],
        legend=False
    )
    ax[2].set_ylabel("Seasonal", fontsize=12)

    # Residual
    sns.lineplot(
        data=df_for_stl_plot,
        x='yearmon',
        y='resid',
        ax=ax[3],
        color=stl_palette[3],
        legend=False
    )
    ax[3].set_ylabel("Residual", fontsize=12)

    # Shared x-axis formatting
    ax[3].set_xlabel("\nYear and Month", fontsize=14)
    # main title above first plot
    ax[0].set_title(f"STL Decomposition for {stl_title_note}\n", fontsize=14)

    year_start_mask = df_for_stl_plot['month'] == 1
    xticks = df_for_stl_plot.loc[year_start_mask, 'yearmon'].drop_duplicates()
    xtick_labels = df_for_stl_plot.loc[year_start_mask, 'year'].drop_duplicates().astype(str)
    ax[3].set_xticks(xticks)
    ax[3].set_xticklabels(xtick_labels) 

    if add_seasonal_plus_mean_trend_to_plot:
        sns.lineplot(
        data=df_for_stl_plot,
        x='yearmon',
        y='seasonal_plus_mean_trend',
        color = stl_palette[4],
        errorbar=None,
        ax=ax[0],
        legend=False
        )
        ax[0].set_ylabel("Data and STL Fit", fontsize=12)
    
    if return_stl_results_df == True:
        return fig, ax, df_for_stl_plot
    else:
        return fig, ax




# %% --------------------------------------------------------------------------
# plot anomaly profiles (year vs. long-running average)
def plot_anomaly_profiles(
        plot_df,
        plot_year = None,
        plot_df_outcome_breaks = None
    ):

    fig, ax = plt.subplots(figsize=(15, 5)) 

    df = plot_df.copy()

    unique_years = sorted(df["year"].unique())
    palette = sns.color_palette("husl", len(unique_years))
    year_color_dict = dict(zip(unique_years, palette))

    if plot_year:
        df = df.loc[df['year'] == plot_year]
        line_color = year_color_dict.get(plot_year, "black")
        sns.lineplot(
            data=df, 
            x='month', 
            y= 'z_anomaly', 
            color = line_color,
            linewidth=2,
            errorbar=None  
        )
        ax.set_xlabel("\nMonth", fontsize=14)
        xticks = df['month'].drop_duplicates().tolist()
        xtick_labels = df['month'].drop_duplicates().tolist()

    else: 
        sns.lineplot(
            data=df, 
            x='month', 
            y= 'z_anomaly', 
            hue='year',
            palette = palette,
            linewidth=2,
            errorbar=None  
        )
        plt.legend(title='Year\n', labels=[str(year) for year in unique_years])
        ax.set_xlabel("\nMonth", fontsize=14)
        ax.legend(
            title="Year\n",
            loc='upper left', 
            bbox_to_anchor=(1.02, 1.0), 
            ncols = 2    
        )
        xticks = df['month'].drop_duplicates().tolist()
        xtick_labels = df['month'].drop_duplicates().tolist()

    # set the ticks for either of the two plot options
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

    # reference line at 0
    ax.axhline(0, color='grey', linestyle='--')

    if plot_df_outcome_breaks is None:
        plot_df_outcome_breaks = get_nice_breaks_from_list_of_df_columns(
            plot_df, 
            ['z_anomaly'], 
            min_value = None, 
            min_breaks=4, 
            max_breaks=8,
        )
    ax.set_yticks(plot_df_outcome_breaks)    
    ax.set_yticklabels(plot_df_outcome_breaks)

    if plot_year:
        ax.set_title(f"Anomaly Profiles by Month for {plot_year}\n", fontsize=14)
    else:
        ax.set_title(f"Anomaly Profiles by Month - Multiple Years\n", fontsize=14)

    ax.set_ylabel("Anomaly Z-Score\n", fontsize=14)

    return fig, ax


# %% --------------------------------------------------------------------------
# plot anomaly extremity profiles (year vs. long-running average, as a category)
def plot_anomaly_extremity_profiles(
    plot_df,
    plot_year=None,
    category_scheme="integer",  # or "percentile"
):

    df = plot_df.copy()

    unique_years = sorted(df["year"].unique())
    palette = sns.color_palette("husl", len(unique_years))
    year_color_dict = dict(zip(unique_years, palette))

    # Mapping categories to values
    if category_scheme == "integer":
        category_map = {
            "extremely low (<=5%)": -3,
            "very low (>5%, <=10%)": -2,
            "low (>10%, <=20%)": -1,
            "normal": 0,
            "high (>=80%, <90%)": 1,
            "very high (>=90%, <95%)": 2,
            "extremely high (>=95%)": 3
        }
        y_ticks = [-3, -2, -1, 0, 1, 2, 3]
    elif category_scheme == "percentile":
        category_map = {
            "extremely low (<=5%)": -95,
            "very low (>5%, <=10%)": -90,
            "low (>10%, <=20%)": -80,
            "normal": 0,
            "high (>=80%, <90%)": 80,
            "very high (>=90%, <95%)": 90,
            "extremely high (>=95%)": 95
        }
        y_ticks = [-95, -90, -80, 0, 80, 90, 95]
    else:
        raise ValueError("category_scheme must be 'integer' or 'percentile'")

    # Add numeric category
    df["category_numeric"] = df["anomaly_category"].map(category_map)

    # Drop missing category mappings if any
    df = df.dropna(subset=["category_numeric"])

    # Plot
    fig, ax = plt.subplots(figsize=(15, 5)) 

    if plot_year:
        df = df[df["year"] == plot_year]
        line_color = year_color_dict.get(plot_year, "black")
        sns.lineplot(
            data=df,
            x="month",
            y="category_numeric",
            linewidth=2,
            color = line_color,
            errorbar=None,
            ax=ax
        )
        ax.set_title(f"Anomaly Extremity Profile by Month for {plot_year}\n", fontsize=14)
    else:
        sns.lineplot(
            data=df,
            x="month",
            y="category_numeric",
            hue="year",
            palette=palette,
            linewidth=2,
            errorbar=None,
            ax=ax
        )
        ax.set_title(f"Anomaly Extremity Profiles by Month - Multiple Years\n", fontsize=14)
        ax.legend(
            title="Year\n", 
            bbox_to_anchor=(1.02, 1.0), 
            loc="upper left", 
            ncols=2
        )

    # Label axes
    ax.set_xlabel("\nMonth", fontsize=14)
    ax.set_ylabel("Anomaly Category\n", fontsize=14)
    ax.set_xticks(range(1, 13))

    # reference line at 0
    ax.axhline(0, color='grey', linestyle='--')

    # Set custom y-axis labels
    reverse_map = {v: k for k, v in category_map.items()}
    y_labels = [reverse_map.get(v, "") for v in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    return fig, ax


# %% --------------------------------------------------------------------------
# map percent or number of years seasonality metric exceeds a particular threshold
def map_percent_or_number_of_years_above_threshold(
    _gdf,                              # '_gdf' (not 'gdf') to flag as unhashable for streamlit st.cache_data decorator...
    column,
    cmap_name='Spectral_r',
    percent_or_number = 'number',
    percent_bin_size = 10,
    title="",
    colorbar_label="Number of Years\n",
    figsize=(16, 9),
    ax=None,
    legend_shrink=0.8,   # shrink factor for colorbar size (1.0 = default)
    edgecolor='black',
    linewidth=0.5,
    additional_boundary_outline_gdf = None,
    additional_boundary_color = 'black',
):
    """

    """

    if percent_or_number == 'percent':
        values = _gdf[column].astype(float)
        vmin = 0.0
        vmax = 100.0
        # add a tiny epsilon to include the top edge
        epsilon = 1e-6
        levels = np.arange(vmin, vmax + percent_bin_size + epsilon, percent_bin_size)

    elif percent_or_number == 'number':
        values = _gdf[column].astype('Int64')
        vmin = 0
        vmax = int(values.max())
        levels = np.arange(vmin, vmax + 2)  # use of +1 would exclude upper edge

    # colormap with as many colors as needed
    cmap_base = plt.colormaps[cmap_name].resampled(len(levels) - 1)
    cmap = ListedColormap(cmap_base(np.arange(cmap_base.N)))
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    # figure & axis
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # plot
    _gdf.plot(
        column=values,
        cmap=cmap,
        norm=norm,
        edgecolor=edgecolor,
        linewidth=linewidth,
        legend=False,
        ax=ax
    )

    # Plot the COUNTRY boundaries over the top in black
    if additional_boundary_outline_gdf is not None:

        additional_boundary_outline_gdf.plot(
            ax=ax,
            facecolor='none',
            edgecolor= additional_boundary_color,
            linewidth=1.5,
            zorder=2
        )

    # add discrete colorbar
    cbar = plt.colorbar(
        cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=ax,
        boundaries=levels,
        ticks=levels[:-1],
        shrink=legend_shrink
    )
    cbar.set_label(colorbar_label, rotation=270, labelpad=15, fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    ax.set_title(title, fontsize=14)
    
    # ax.axis('off')
    
    return fig, ax
