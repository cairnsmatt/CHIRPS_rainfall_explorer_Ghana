# %% --------------------------------------------------------------------------
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import seaborn as sns

from matplotlib.colors import BoundaryNorm, ListedColormap

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances



# %% --------------------------------------------------------------------------
def plot_climatological_water_season_for_selected_area(
    anomaly_df, 
    area_column, 
    selected_area,
    save_path = None
):

    plot_area_mask = anomaly_df[area_column] == selected_area 
    plot_df = anomaly_df.loc[plot_area_mask].copy()

    if plot_df.empty:
        print(f"Warning: No data found for selected area '{selected_area}'. Skipping plot generation.")
        return 

    # build a min_max_df for scatterplot at minimum, maximum
    try:
        max_row = plot_df.loc[plot_df['cumulative_daily_mean_anomaly'].idxmax()]
        min_row = plot_df.loc[plot_df['cumulative_daily_mean_anomaly'].idxmin()]
        min_max_df = pd.concat([max_row.to_frame().T, min_row.to_frame().T], ignore_index=True)
        min_max_df['doy'] = min_max_df['doy'].astype(int)
        min_max_df['cumulative_daily_mean_anomaly'] = min_max_df['cumulative_daily_mean_anomaly'].astype(float)
    except ValueError as e:
        print(f"Error processing min/max for '{selected_area}': {e}. Plotting lines only.")
        min_max_df = pd.DataFrame() # Create an empty DataFrame to skip scatterplot

    # Plot Climatological Mean Precip, Daily Mean Anomaly, Cumulative
    # x axis, lh y axis
    fig, ax1 = plt.subplots(figsize=(16, 9))

    palette = sns.color_palette('muted')
    color1 = palette[3] # red
    color2 = palette[0] # blue
    color3 = palette[2] # green
    color4 = palette[6] # pink

    sns.lineplot(
        data=plot_df,
        x='doy',
        y='mean_day_i',
        linewidth=2,
        errorbar=None,
        ax=ax1, # Plot mean_day_i on ax1
        color=color1,
        label='Climatological mean precipitation in mm',
        legend=False
    )
    sns.lineplot(
        data=plot_df,
        x='doy',
        y='daily_mean_anomaly',
        linewidth=2,
        errorbar=None,
        ax=ax1, # Plot mean_day_i on ax1
        color=color2,
        label='Daily mean precipitation anomaly in mm',
        legend=False
    )
    ax1.set_ylabel("Mean Precipitation (mm), Daily Mean Precipitation Anomaly (mm)\n", fontsize=14)
    ax1.tick_params(axis='both', labelsize=12)

    ax1.set_xlabel("\n Day of the Year", fontsize=14) # X-axis label goes on the primary axis

    # rh axis linked to same x axis
    ax2 = ax1.twinx() 

    # --- SECONDARY AXIS (cumulative_daily_mean_anomaly) ---
    sns.lineplot(
        data=plot_df,
        x='doy',
        y='cumulative_daily_mean_anomaly',
        linewidth=2,
        errorbar=None,
        ax=ax2, # Plot cumulative_daily_mean_anomaly on ax2
        color=color3,
        label='Cumulative daily mean precipitation anomaly',
        legend=False
    )
    ax2.set_ylabel(f"\nCumulative Daily Mean Precipitation Anomaly", fontsize=14)
    ax2.tick_params(axis='y', labelsize=12)

    sns.scatterplot(
        data=min_max_df,
        x='doy',
        y='cumulative_daily_mean_anomaly',
        ax=ax2,
        edgecolor = 'black',
        color=color4, 
        s=300,            # (size in points^2)
        marker='o',       # Circle marker
        zorder=5,         # Ensure markers are on top of the line
        legend=False
    )

    # Title and Legend (placed on ax1 for central control)
    plt.title(f"{selected_area}\n", fontsize=16, ha='center')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi = 300, bbox_inches='tight')

    return fig, ax1, ax2


# %% --------------------------------------------------------------------------
# plot yearly cumulative 
def plot_yearly_anomaly_for_selected_area(
    yearly_anomaly_df,
    area_column,
    selected_area,
    save_path = None
):
    
    plot_district_mask = yearly_anomaly_df[area_column] == selected_area 
    plot_df = yearly_anomaly_df.loc[plot_district_mask].copy()
    plot_df = plot_df.sort_values('year')

    if plot_df.empty:
        print(f"Warning: No data found for selected area '{selected_area}'. Skipping plot generation.")
        return 

    # x axis, lh y axis
    fig, ax = plt.subplots(figsize=(16, 9))
    unique_years = plot_df['year'].unique()
    years_palette = sns.color_palette('husl', len(unique_years))

    sns.lineplot(
        data=plot_df,
        x='doy',
        y='daily_cumulative_rainfall_anomaly',
        linewidth=2,
        alpha=0.7,
        errorbar=None,
        ax=ax, 
        hue='year',
        palette=years_palette
    )
    ax.set_ylabel("Daily Cumulative Precipitation Anomaly\n", fontsize=14)
    ax.set_xlabel("\nDay of the Year", fontsize=14)

    plt.legend(title='Year\n', labels=[str(year) for year in unique_years])
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), title="Year\n")
    ax.tick_params(axis='both', labelsize=12)

    # Title and Legend (placed on ax1 for central control)
    plt.title(f"Daily Cumulative Precipitation Anomaly - {selected_area}\n", fontsize=16, ha='center')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi = 300, bbox_inches='tight')

    return fig, ax


# %% --------------------------------------------------------------------------
# plot yearly water season extent for selected area 
def plot_yearly_water_season_for_selected_area(
    yearly_water_season_df,
    area_column,
    selected_area,
    save_path = None
):
    
    plot_district_mask = yearly_water_season_df[area_column] == selected_area 
    plot_df = yearly_water_season_df.loc[plot_district_mask].copy()
    plot_df = plot_df.sort_values('year')

    if plot_df.empty:
        print(f"Warning: No data found for selected area '{selected_area}'. Skipping plot generation.")
        return 

    min_year = plot_df['year'].min()
    max_year = plot_df['year'].max()

    # x axis, lh y axis
    fig, ax = plt.subplots(figsize=(16, 9))
    unique_years = plot_df['year'].unique()
    years_palette = sns.color_palette('husl', len(unique_years))
    year_to_color = dict(zip(unique_years, years_palette))

    # Plot horizontal lines (one per year)
    for _, row in plot_df.iterrows():
        ax.hlines(
            y=row['year'],
            xmin=row['onset_doy'],
            xmax=row['cessation_doy'],
            color=year_to_color[row['year']],
            linewidth=4,
            alpha=0.9
        )

    # Axis labels and limits
    ax.set_xlabel("\nDay of Year (DOY)\n", fontsize=14)
    ax.set_ylabel("Year\n", fontsize=14)
    ax.set_ylim(max_year+1, min_year-1)
    ax.set_xlim(0, 365)
    ax.tick_params(axis='both', labelsize=12)

    # Create a custom legend with year colors
    handles = [
        plt.Line2D([0], [0], color=year_to_color[y], lw=4, label=str(y))
        for y in unique_years
    ]
    ax.legend(handles=handles, title="Year\n", bbox_to_anchor=(1.02, 1.0), loc="upper left")

    # Title and style
    plt.title(f"\nWater Season Duration — {selected_area}\n\n", fontsize=16, ha='center')
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi = 300, bbox_inches='tight')

    return fig, ax


# %% --------------------------------------------------------------------------
# get boundaries, cmap norm for onset cessation map

def get_boundaries_cmap_norm_for_onset_cessation_colour_scheme(gdf, legend_interval, colour_scheme, min_onset = None, max_cessation = None):

    gdf = gdf.copy()

    if min_onset is None:
        min_onset = gdf['onset_doy'].min()
    if max_cessation is None:
        max_cessation = gdf['cessation_doy'].max()

    min_break = (np.floor(min_onset/legend_interval))*legend_interval
    max_break = (np.ceil(max_cessation/legend_interval))*legend_interval 
    boundaries = np.arange(min_break, max_break, legend_interval)

    # Determine the number of discrete colors needed
    # The number of colors is one less than the number of boundaries (i.e., the number of bins)
    n_bins = len(boundaries) - 1

    # Generate the HUSL color palette with the required number of discrete steps (n_bins)
    chosen_colors = sns.color_palette(colour_scheme, n_bins)

    # Create a custom discrete colormap object
    cmap = ListedColormap(chosen_colors)

    # Create the normalization object (using the new map's color count)
    # cmap_husl.N is the number of colors in the map (which is n_bins)
    norm = colors.BoundaryNorm(boundaries, cmap.N)

    return boundaries, cmap, norm


# %% --------------------------------------------------------------------------
# map onset or cessation date

def map_onset_or_cessation_from_water_season_gdf(
        gdf,
        onset_or_cessation = 'onset',
        legend_interval = 10,
        colour_scheme = 'Spectral',
        min_onset = None,
        max_cessation = None,
        additional_boundary_outline_gdf = None,
        additional_boundary_color = 'black',
        save_path = None
):

    boundaries, cmap, norm = get_boundaries_cmap_norm_for_onset_cessation_colour_scheme(
        gdf, 
        legend_interval, 
        colour_scheme,
        min_onset=min_onset,
        max_cessation=max_cessation
    )

    fig, ax = plt.subplots(figsize=(16, 9))
    
    if onset_or_cessation == 'onset':
        ax.set_title(f'Rainfall Onset ({legend_interval}-Day Intervals)\n', fontsize=16)

        # Plotting the raw 'onset_doy' column, but using the discrete 'norm'
        gdf.plot(
            column='onset_doy',     # Plot the original DOY column
            cmap=cmap,
            norm=norm,              # applies the binning to the color scale
            legend=True,
            legend_kwds={
                'label': f"\nDay of Year",
                'orientation': "vertical",
                'ticks': boundaries[1::2] # Show every other bin boundary
            },
            linewidth=0.5,
            ax=ax
        )

    elif onset_or_cessation == 'cessation':

        ax.set_title(f'Rainfall Cessation ({legend_interval}-Day Intervals)\n', fontsize=16)

        # Plotting the raw 'onset_doy' column, but using the discrete 'norm'
        gdf.plot(
            column='cessation_doy',     # Plot the original DOY column
            cmap=cmap,
            norm=norm,              # applies the binning to the color scale
            legend=True,
            legend_kwds={
                'label': f"\nDay of Year",
                'orientation': "vertical",
                'ticks': boundaries[1::2] # Show every other bin boundary
            },
            ax=ax
        )

    else: 
        print("choose either 'onset' or 'cessation' as option")

    # flag areas with a biannual season in grey:
    biannual = gdf[gdf['seasonality_regime'] == 'biannual']
    if not biannual.empty:
        biannual.plot(ax=ax, color='lightgrey', edgecolor='none', zorder=2)

    # Plot the COUNTRY boundaries over the top in black
    if additional_boundary_outline_gdf is not None:

        additional_boundary_outline_gdf.plot(
            ax=ax,
            facecolor='none',
            edgecolor= additional_boundary_color,
            linewidth=0.5,
            zorder=3
        )

    # remove black border around map
    # ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi = 300, bbox_inches='tight')

    return fig, ax



# %% --------------------------------------------------------------------------
def map_water_season_length_from_gdf(
        gdf,
        legend_interval = 10,
        colour_scheme = 'Spectral',
        additional_boundary_outline_gdf = None,
        save_path = None
):

    boundaries, cmap, norm = get_boundaries_cmap_norm_for_onset_cessation_colour_scheme(
        gdf, 
        legend_interval, 
        colour_scheme
    )

    fig, ax = plt.subplots(figsize=(16, 9))
    
    gdf = gdf.copy()
    gdf['water_season_length'] = gdf['cessation_doy'] - gdf['onset_doy']

    ax.set_title(f'Water Season Length ({legend_interval}-Day Intervals)\n', fontsize=16)

    # Plotting the raw 'onset_doy' column, but using the discrete 'norm'
    gdf.plot(
        column='water_season_length',     # Plot the original DOY column
        cmap=cmap,
        norm=norm,              # applies the binning to the color scale
        legend=True,
        legend_kwds={
            'label': f"\nWater Season Length (Days)",
            'orientation': "vertical",
            'ticks': boundaries[1::2] # Show every other bin boundary
        },
        linewidth=0.5,
        ax=ax
    )

    # Plot the COUNTRY boundaries over the top in black
    if additional_boundary_outline_gdf is not None:

        additional_boundary_outline_gdf.plot(
            ax=ax,
            facecolor='none',
            edgecolor='black',
            linewidth=0.5,
            zorder=2
        )

    if save_path:
        plt.savefig(save_path, dpi = 300, bbox_inches='tight')

    return fig, ax



# %% --------------------------------------------------------------------------
# get plotting grid from shapefile
def define_geofacet_grid_from_shapefile_centroids(
    gdf,
    area_col,
    max_diff=4,
    tweak_wider=0,
    tweak_longer=0
):
    """
    Assign areas to a spatially coherent grid layout.

    Parameters:
    - gdf: GeoDataFrame with polygon geometries.
    - area_col: column with area names or IDs.
    - max_diff: max allowed difference between rows and cols in candidate grids.
    - tweak_wider: int, number of additional columns to make grid wider.
    - tweak_longer: int, number of additional rows to make grid longer.

    Returns:
    - DataFrame with [area_col, row, col]
    """

    if gdf.crs is None or not gdf.crs.is_projected:
        gdf = gdf.to_crs(epsg=3857)

    areas = gdf[area_col].values
    n = len(areas)

    centroids = gdf.geometry.centroid
    coords = np.column_stack((centroids.x, centroids.y))

    # Normalize to [0, 1] to compare with grid positions
    coords_norm = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0))

    def generate_candidate_grids(n, max_diff):
        grids = []
        for cols in range(1, n + 1):
            rows = int(np.ceil(n / cols))
            if abs(rows - cols) <= max_diff:
                grids.append((rows, cols))
        return grids

    def identify_best_layout(coords_norm, grids, tweak_wider, tweak_longer):
        best_score = np.inf
        best_layout = None
        best_assignment = None

        for base_rows, base_cols in grids:
            rows = base_rows + tweak_longer
            cols = base_cols + tweak_wider

            if rows * cols < n:
                continue  # skip if not enough cells

            xg, yg = np.meshgrid(np.linspace(0, 1, cols), np.linspace(0, 1, rows))
            grid_coords = np.column_stack((xg.ravel(), yg.ravel()))

            dists = pairwise_distances(coords_norm, grid_coords)

            row_ind, col_ind = linear_sum_assignment(dists)

            total_dist = dists[row_ind, col_ind].sum()

            if total_dist < best_score:
                best_score = total_dist
                best_layout = (rows, cols)
                best_assignment = col_ind

        return best_layout, best_assignment

    grids = generate_candidate_grids(n, max_diff)
    if not grids:
        raise ValueError("No candidate grids found. Try increasing max_diff.")

    (n_rows, n_cols), assignment = identify_best_layout(coords_norm, grids, tweak_wider, tweak_longer)

    # Get final grid positions
    xg, yg = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
    yg = yg[::-1]  # flip rows so north is at the top
    grid_positions = np.column_stack((xg.ravel(), yg.ravel()))

    assigned_positions = grid_positions[assignment]

    area_grid_df = pd.DataFrame({
        'area': areas,
        'row': assigned_positions[:, 1],
        'col': assigned_positions[:, 0],
    }).astype({ 'row': int, 'col': int })

    return area_grid_df


# %% --------------------------------------------------------------------------
# TODO - finish adapting this code from LSTM use case for daily rainfall
# Plot all areas on grid 
def plot_climatological_water_season_on_geofacet_grid(
        anomaly_df,
        area_grid_df,
        area_column,
        save_path = None,
        share_y_axis_across_grid = True,
        axis_and_subtitle_fontsize = 8,
        subtitle_inside_plot_area = True,
        print_mismatches_between_dfs = False
    ):

    nrows = area_grid_df['row'].max() + 1
    ncols = area_grid_df['col'].max() + 1    

    anomaly_df = anomaly_df.copy()
    anomaly_df['area'] = anomaly_df[area_column]

    # merge the two dfs
    plot_df = pd.merge(anomaly_df, area_grid_df, how = 'left', on = 'area')
    plot_df.sort_values(by=['area', 'doy'], inplace=True)

    # optionally, print mismatches between dfs
    if print_mismatches_between_dfs:
        pred_areas = set(anomaly_df['area'].astype(str).unique())
        grid_areas = set(area_grid_df['area'].astype(str).unique())

        missing_in_area_grid_df = sorted(pred_areas - grid_areas)
        if missing_in_area_grid_df:
            print(f"\nThe following areas are in anomaly_df but not in area_grid_df:\n")
            for area in missing_in_area_grid_df:
                print(area)
            print("\n")

        missing_in_anomaly_df = sorted(grid_areas - pred_areas)
        if missing_in_anomaly_df:
            print(f"The following areas are in area_grid_df but not in anomaly_df:\n")
            for area in missing_in_anomaly_df:
                print(area)
            print("\n")




    # Create the figure and axes for subplots
    fig, axes = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(16, 9), 
        sharex=True, 
        sharey=share_y_axis_across_grid
    )

    palette = sns.color_palette('bright')
    color1 = palette[3] # red
    color2 = palette[0] # blue
    color3 = palette[2] # green

    # if share_y_axis across all plots, get y breaks outside the loop
    if share_y_axis_across_grid == True:
        y_breaks = get_nice_breaks_from_list_of_df_columns(
            plot_df, 
            ['mean_day_i', 'daily_mean_anomaly'], 
            min_value = 0, 
            min_breaks=3, 
            max_breaks=5
        )

    # Loop through each subplot position
    for idx, row in area_grid_df.iterrows():
        ax = axes[row['row'], row['col']]  # Access subplot at (row, col) position
        chosen_district = row['area']

        # verify chosen district is present
        if chosen_district not in anomaly_df['area'].unique():  
            print(f"Area '{chosen_district}' exists in area_grid_df but is missing from anomaly_df.")
            continue

        # Filter the dataset for the current area and verify not empty (if it is, skip to next)
        subset_df = plot_df[plot_df['area'] == chosen_district].copy()
        if subset_df.empty:
            ax.axis('off')
            continue

        # if each plot gets its own y axis, define the breaks within the loop:
        if share_y_axis_across_grid == False:
            y_breaks = get_nice_breaks_from_list_of_df_columns(
                subset_df, 
                ['mean_day_i', 'daily_mean_anomaly'], 
                min_value = 0, 
                min_breaks=3, 
                max_breaks=5
            )

        sns.lineplot(
            data=plot_df,
            x='doy',
            y='mean_day_i',
            linewidth=2,
            errorbar=None,
            ax=ax, # Plot mean_day_i on ax1
            color=color1,
            label='climatological mean precipitation (day i) in mm',
            legend=False
        )
        sns.lineplot(
            data=plot_df,
            x='doy',
            y='daily_mean_anomaly',
            linewidth=2,
            errorbar=None,
            ax=ax, # Plot mean_day_i on ax1
            color=color2,
            label='daily mean rainfall anomaly (day i) in mm',
            legend=False
        )

        # rh axis linked to same x axis
        ax2 = ax.twinx() 

        # --- SECONDARY AXIS (cumulative_daily_mean_anomaly) ---
        sns.lineplot(
            data=plot_df,
            x='doy',
            y='cumulative_daily_mean_anomaly',
            linewidth=2,
            errorbar=None,
            ax=ax2, # Plot cumulative_daily_mean_anomaly on ax2
            color=color3,
            label='cumulative daily mean precipitation anomaly',
            legend=False
        )

        ###########################################################################

        ax.set_ylabel("")
        # ax.set_ylim(bottom=0, top=y_breaks[-1]) 
        ax.set_yticks(y_breaks)    
        ax.set_yticklabels(y_breaks, fontsize=axis_and_subtitle_fontsize)

        ax.set_xlabel("")
        # xticks = np.arange(0, len(subset_df), months_between_xticks)

        if subtitle_inside_plot_area:
            ax.set_title("")
            ax.text(0.5, 0.95, f"{chosen_district}", transform=ax.transAxes,  # last arg means 'relative to axes'
            fontsize=axis_and_subtitle_fontsize, color='grey',
            verticalalignment='top', horizontalalignment='center')
        else:
            ax.set_title(f"{chosen_district}", fontsize = axis_and_subtitle_fontsize)

        # # Set y-ticks every 100
        # ax.set_yticks(np.arange(0, ax.get_ylim()[1], 100))

    ###########################################################################
    # Suppress blank grids 
    for row in range(nrows):
        for col in range(ncols):
            if not any(
                (area_grid_df['row'] == row) & 
                (area_grid_df['col'] == col)
            ):
                axes[row, col].axis('off')  # Hide subplot        
    ###########################################################################
    # Adjust Subplot Parameters
    plt.subplots_adjust(left=0.075, right=0.95, bottom=0.1, top=0.9, wspace=0.1, hspace=0.3)
    
    # Add elements to the "rest" of the Figure
    # # Legend outside the plot
    # fig.legend(handles=handles, loc='lower right', bbox_to_anchor=(1.0, -0.02), title="", ncol=2)
    
    # # Common y-axis title (on the left side)
    # if outcome_label_for_plots:
    #     fig.text(0.04, 0.5, f"{outcome_label_for_plots}\n", va='center', ha='center', rotation='vertical', fontsize=14)

    # Common x-axis title (Bottom, Centred)
    fig.text(0.5, 0.02, 'Year', va='bottom', ha='center', fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi = 300, bbox_inches='tight')

    plt.show()     


# %% --------------------------------------------------------------------------
# TODO - finish adapting this code from LSTM use case for daily rainfall
# Plot all areas on grid 
def plot_yearly_water_season_on_geofacet_grid(
        yearly_water_season_df,
        area_grid_df,
        area_column,
        save_path = None,
        axis_and_subtitle_fontsize = 8,
        subtitle_inside_plot_area = False,
        print_mismatches_between_dfs = False
    ):

    nrows = area_grid_df['row'].max() + 1
    ncols = area_grid_df['col'].max() + 1    

    yearly_water_season_df = yearly_water_season_df.copy()
    yearly_water_season_df['area'] = yearly_water_season_df[area_column]

    # merge the two dfs
    plot_df = pd.merge(yearly_water_season_df, area_grid_df, how = 'left', on = 'area')
    plot_df.sort_values(by=['area', 'year'], inplace=True)

    # optionally, print mismatches between dfs
    if print_mismatches_between_dfs:
        pred_areas = set(yearly_water_season_df['area'].astype(str).unique())
        grid_areas = set(area_grid_df['area'].astype(str).unique())

        missing_in_area_grid_df = sorted(pred_areas - grid_areas)
        if missing_in_area_grid_df:
            print(f"\nThe following areas are in yearly_water_season_df but not in area_grid_df:\n")
            for area in missing_in_area_grid_df:
                print(area)
            print("\n")

        missing_in_yearly_water_season_df = sorted(grid_areas - pred_areas)
        if missing_in_yearly_water_season_df:
            print(f"The following areas are in area_grid_df but not in yearly_water_season_df:\n")
            for area in missing_in_yearly_water_season_df:
                print(area)
            print("\n")


    # Create the figure and axes for subplots
    fig, axes = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(16, 12), 
        sharex=True, 
        sharey=True
    )

    unique_years = plot_df['year'].unique()
    years_palette = sns.color_palette('husl', len(unique_years))
    year_to_color = dict(zip(unique_years, years_palette))


    # Loop through each subplot position
    for idx, row in area_grid_df.iterrows():
        ax = axes[row['row'], row['col']]  # Access subplot at (row, col) position
        chosen_district = row['area']

        # verify chosen district is present
        if chosen_district not in yearly_water_season_df['area'].unique():  
            print(f"Area '{chosen_district}' exists in area_grid_df but is missing from yearly_water_season_df.")
            continue

        # Filter the dataset for the current area and verify not empty (if it is, skip to next)
        subset_df = plot_df[plot_df['area'] == chosen_district].copy()
        if subset_df.empty:
            ax.axis('off')
            continue

        # Plot horizontal lines (one per year)
        for _, row in subset_df.iterrows():
            ax.hlines(
                y=row['year'],
                xmin=row['onset_doy'],
                xmax=row['cessation_doy'],
                color=year_to_color[row['year']],
                linewidth=0.5,
                alpha=0.9
            )

        ###########################################################################


        ax.set_xlabel("")
        # xticks = np.arange(0, len(subset_df), months_between_xticks)

        if subtitle_inside_plot_area:
            ax.set_title("")
            ax.text(0.5, 0.95, f"{chosen_district}", transform=ax.transAxes,  # last arg means 'relative to axes'
            fontsize=axis_and_subtitle_fontsize, color='grey',
            verticalalignment='top', horizontalalignment='center')
        else:
            ax.set_title(f"{chosen_district}", fontsize = axis_and_subtitle_fontsize)


    ###########################################################################
    # Suppress blank grids 
    for row in range(nrows):
        for col in range(ncols):
            if not any(
                (area_grid_df['row'] == row) & 
                (area_grid_df['col'] == col)
            ):
                axes[row, col].axis('off')  # Hide subplot        
    ###########################################################################
    # Adjust Subplot Parameters
    plt.subplots_adjust(left=0.075, right=0.95, bottom=0.1, top=0.9, wspace=0.1, hspace=0.3)
    
    # Add elements to the "rest" of the Figure
    # # Legend outside the plot
    # fig.legend(handles=handles, loc='lower right', bbox_to_anchor=(1.0, -0.02), title="", ncol=2)
    
    # # Common y-axis title (on the left side)
    # if outcome_label_for_plots:
    #     fig.text(0.04, 0.5, f"{outcome_label_for_plots}\n", va='center', ha='center', rotation='vertical', fontsize=14)

    # Common x-axis title (Bottom, Centred)
    fig.text(0.5, 0.02, 'Year', va='bottom', ha='center', fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi = 300, bbox_inches='tight')

    plt.show()     




# %% --------------------------------------------------------------------------
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


