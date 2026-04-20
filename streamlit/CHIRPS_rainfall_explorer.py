# %%
# Because plots and metrics are NOT a subfolder of streamlit (they are in src)
# Ensure project root is in Python's module search path
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# %%
# custom imports
from src.seasonality_metrics_plots.seasonality_plotting_functions import (  
    get_nice_breaks_from_list_of_df_columns,
    plot_monthly_data_with_optional_year_overlay_or_second_outcome,
    plot_percentage_of_annual_totals_by_month_or_in_windows,
    plot_start_month_consistency_histogram_using_year_summary_df,
    map_percent_or_number_of_years_above_threshold
)

from src.CHIRPS_daily_data.daily_plotting_functions import (
    plot_climatological_water_season_for_selected_area,
    plot_yearly_water_season_for_selected_area,
    map_onset_or_cessation_from_water_season_gdf
)

from streamlit_helper_functions import (
    apply_decorator, 
    load_and_return_df,
    polars_load_admin1_and_area_time_series_adding_yearmon,
    polars_load_df_and_filter_to_selected_admin1_and_area,
    polars_get_two_row_min_max_df,
    polars_load_threshold_summary_df_filtered_by_window_size_and_threshold_percentage,
    create_gdf_from_shapefile_path,
    make_plot_download_button, 
    make_df_download_button
)


#%%
# imports
import io
import os
import numpy as np
import pandas as pd
import geopandas as gpd

import polars as pl

import streamlit as st
import folium
from streamlit_folium import st_folium


# %%
# define a streamlit decorator and apply this to the imported functions

streamlit_decorator = st.cache_data
apply_streamlit_decorator_flag = True

functions_to_decorate = [
    # seasonality_plotting_functions.py
    "plot_monthly_data_with_optional_year_overlay_or_second_outcome",
    "plot_percentage_of_annual_totals_by_month_or_in_windows",
    "plot_start_month_consistency_histogram_using_year_summary_df",
    # DO NOT DECORATE "map_percent_or_number_of_years_above_threshold" 
    # (this prevents country updating when changed)
    
    # daily_plotting_functions
    "plot_climatological_water_season_for_selected_area",
    "plot_yearly_water_season_for_selected_area"
]

for name in functions_to_decorate:
    globals()[name] = apply_decorator(
        streamlit_decorator, 
        apply_streamlit_decorator_flag
    )(globals()[name])



# %%
# options
area_column = 'NAME_2'
admin1_column = 'NAME_1'
area_column_list = [admin1_column, area_column]

country_dictionary = {
    "Ghana" : "GHA",
}

outcome_dictionary = {
    'Total Monthly Rainfall in mm': 'total_rainfall'  
}


# %%
# Directories - assuming streamlit is at first level below project (and sibling to project/src)
# streamlit_folder = os.getcwd()

# project_folder = os.path.dirname(streamlit_folder)

project_folder = str(project_root)
src_folder = os.path.join(project_folder, 'src')
processed_data_folder = os.path.join(project_folder, 'data', 'processed', 'CHIRPS_monthly_data')
seasonality_metrics_folder = os.path.join(project_folder, 'data', 'processed', 'CHIRPS_monthly_seasonality_metrics')
climatology_data_folder = os.path.join(project_folder, 'data', 'processed', 'CHIRPS_daily_climatology')
shapefile_folder = os.path.join(project_folder, 'shapefiles')




# %%
# streamlit start up 

# overall app width
st.set_page_config(layout="wide")

# Initialize session state
if 'selected_area' not in st.session_state:
    st.session_state.selected_area = "-- Select an area --" 
if 'selected_admin1' not in st.session_state:
    st.session_state.selected_admin1 = "-- Select an admin1 --" 

if 'area_selectbox' not in st.session_state:
    st.session_state.area_selectbox = "-- Select an area --"
if 'area_dropdown_changed' not in st.session_state:
    st.session_state.area_dropdown_changed = False 


# %% --------------------------------------------------------------------------
# define containers and tabs to control rendering behaviour of each section
(   
    country_and_outcome_tab,
    map_tab,
    time_series_tab,
    seasonality_metrics_tab,
    threshold_map_tab,
    climatology_tab,
    rainfall_onset_tab
) = st.tabs([
        'Country and Outcome',
        'Select an Area',
        'Time Series', 
        'Seasonality Metrics', 
        'Map Seasonality Criterion', 
        'Climatology and Water Season',
        'Rainfall Onset/Cessation',
    ])

# %%
with country_and_outcome_tab:

    # 1) Country selection
    st.subheader("Choose a Country - (Please choose Ghana for this Demo)")
    country_options = ["-- Select a country --"] + list(country_dictionary.keys())
    selected_country = st.selectbox(
        "Select a country:", 
        country_options,
        key = 'country_selectbox'

    )
    country_prefix = country_dictionary.get(selected_country) # Use .get for safe dictionary access

    # Reset area selection whenever country changes 
    if "previous_country" not in st.session_state:
        st.session_state.previous_country = selected_country

    if selected_country != st.session_state.previous_country:
        # Country changed → reset area selection states
        st.session_state.selected_area = "-- Select an area --"
        st.session_state.area_selectbox = "-- Select an area --"
        st.session_state.area_dropdown_changed = False

        # reset admin1 
        st.session_state.selected_admin1 = "-- Select an admin1 --"

        # Update stored previous country
        st.session_state.previous_country = selected_country
    

    # Advisory message if no country selected (or if country reset to default)
    if selected_country in (None, "-- Select a country --"):
        st.info("Please select a country to proceed.")
        st.stop()

    #------------------------------------------------------------------------------
    # Data check
    data_df_filename = f"{country_prefix}_monthly_CHIRPS_1995_2024.csv"
    data_df_path = os.path.join(processed_data_folder, data_df_filename)

    if not os.path.exists(data_df_path):
        st.error(f"Data file not found at {data_df_path}")
        st.stop()


    # %%
    # 2) Outcome selection
    st.subheader("Choose Outcome (Please choose Total Monthly Rainfall for this Demo)")

    outcome_options = ["-- Select an outcome --"] + list(outcome_dictionary.keys())
    outcome_selection = st.selectbox(
        "Select an outcome:", 
        outcome_options,
        key = 'outcome_selectbox'
    )

    # Guard clause if placeholder or invalid selection
    if outcome_selection == "-- Select an outcome --":
        outcome_1_column = None
        st.info("Please select an outcome to proceed.")
        st.stop()

    outcome_1_column = outcome_dictionary[outcome_selection]
    outcome_1_label = outcome_selection
    selected_outcome_columns = [outcome_1_column]    


    if outcome_1_column in (None, "[placeholder]"):
        st.info("Please select an outcome to proceed.")
        st.stop()



###########################################################################
# open shapefile for map 
# - decorator used in helpers to avoid reloading
# - .simplify() method used to simplify geometry
country_shapefile_path = os.path.join(
    shapefile_folder, 
    country_prefix, 
    f"gadm41_{country_prefix}_2.shp"
)

(
    gdf, 
    gdf_proj, 
    centroids, 
    centroids_latlon, 
    mean_lat, 
    mean_lon
) = create_gdf_from_shapefile_path(country_shapefile_path, area_column)


###########################################################################
# build country map

# NB. this can't be decorated 

m = folium.Map(location=[mean_lat, mean_lon],
            zoom_start=6, tiles="cartodb positron")

folium.GeoJson(
    gdf,
    name="geojson",
    style_function=lambda feature: {"fillOpacity": 0.5, "weight": 1, "color": "#0080FF"},
    highlight_function=lambda x: {"weight": 3, "color": "#c10534"},
    tooltip=folium.features.GeoJsonTooltip(
        fields=[area_column],
    )
).add_to(m)


###########################################################################
# control aspect ratios for map plotting

col1_relative_width = 8
col2_relative_width = 6
spacer_col_relative_width = 1
col1_scaler = 1.0
col1_width_in_pixels = (1920 / (col1_relative_width + col2_relative_width))*col1_relative_width*col1_scaler
col1_height_in_pixels = (1080 / (col1_relative_width + col2_relative_width))*col1_relative_width*col1_scaler
# # col1_height= col1_width, to make square
# col1_height_in_pixels = col1_width_in_pixels
col2_scaler = 0.8
col2_width_in_pixels = (1920 / (col1_relative_width + col2_relative_width))*col2_relative_width*col2_scaler
col2_height_in_pixels = (1080 / (col1_relative_width + col2_relative_width))*col2_relative_width*col2_scaler


###########################################################################
# Define area_lookup_df and area_display_to_values, used in dropdown

area_lookup_df = (
    gdf[[area_column, admin1_column]]
    .drop_duplicates()
    .sort_values([area_column, admin1_column])
)

area_display_to_values = {
    f"{row[area_column]} ({row[admin1_column]})": (
        row[admin1_column],
        row[area_column]
    )
    for _, row in area_lookup_df.iterrows()
}

###########################################################################
# Define callback for area_selectbox (now returns admin1 as well as name_2)
def update_area_from_dropdown():
    selection = st.session_state.area_selectbox

    st.session_state.area_dropdown_changed = True

    if selection == "-- Select an area --":
        st.session_state.selected_area = "-- Select an area --"
        st.session_state.selected_admin1 = "-- Select an admin1 --"
        return

    admin1_value, area_value = area_display_to_values[selection]

    st.session_state.selected_area = area_value
    st.session_state.selected_admin1 = admin1_value



###########################################################################
# first tab, first expander - map and dropdown for area selection
with map_tab:
        
    # define columns
    col1, spacer_col, col2 = st.columns([
        col1_relative_width, 
        spacer_col_relative_width, 
        col2_relative_width
    ])

    with col1:
        st.subheader(f"Select an area within {selected_country} on the map")

        # 1. Map click selection
        map_data = st_folium(
            m, 
            height= col1_height_in_pixels, 
            width = col1_width_in_pixels, 
            returned_objects=["last_active_drawing"],
            key="folium_map"
        )

        # get the properties (admin1 and area) from the map 
        area_from_map_properties = None
        admin1_from_map_properties = None

        if (
            map_data["last_active_drawing"]
            and isinstance(map_data["last_active_drawing"], dict)
            and "properties" in map_data["last_active_drawing"]
        ):
            properties = map_data["last_active_drawing"]["properties"]
            area_from_map_properties = properties.get(area_column)
            admin1_from_map_properties = properties.get(admin1_column)

        # update the session state variables if map clicked 
        # (if dropdown has not changed, but there is a difference in either
        #  area name or admin1 name vs. the session state variables)
        if not st.session_state.get('area_dropdown_changed', False):
            if (
                area_from_map_properties
                and (
                    area_from_map_properties != st.session_state.selected_area
                    or admin1_from_map_properties != st.session_state.selected_admin1
                )
            ):
                st.session_state.selected_area = area_from_map_properties
                st.session_state.selected_admin1 = admin1_from_map_properties

                display_value = f"{area_from_map_properties} ({admin1_from_map_properties})"
                st.session_state.area_selectbox = display_value
        
        # Finally, reset the flag after checking it.
        # (ensures the map logic is re-enabled on the next run)
        st.session_state.area_dropdown_changed = False


    with spacer_col:
        pass

    with col2:
        st.subheader("Or choose from the list")

        area_options = ["-- Select an area --"] + list(area_display_to_values.keys())

        # render dropdown
        # 'key' links st.session_state.area_selectbox.
        # 'on_change' triggers callback function.
        selected_area_from_dropdown = st.selectbox(
            "Select an area:",
            area_options,
            key="area_selectbox",
            on_change=update_area_from_dropdown 
        )

           

###########################################################################
# Time Series Plots (left - time series, right - overlay)

with time_series_tab:

    # Only make these plots if a valid area is selected (not the placeholder)
    if st.session_state.selected_area != "-- Select an area --":

        time_series_df = polars_load_admin1_and_area_time_series_adding_yearmon(
            data_df_path=data_df_path, 
            admin1_column=admin1_column,
            area_column=area_column,
            selected_admin1=st.session_state.selected_admin1,
            selected_area=st.session_state.selected_area
        )

        if not time_series_df.empty:

            st.subheader(f"Time Series Data for '{st.session_state.selected_area} - ({st.session_state.selected_admin1})'")
            
            ########################################################################
            st.markdown("### Select Year Range for the Plots")

            # build year slider based years in on the df:
            year_min = int(time_series_df["year"].min())
            year_max = int(time_series_df["year"].max())

            time_series_year_range = st.slider(
                "Select year range:",
                min_value=year_min,
                max_value=year_max,
                value=(year_min, year_max),
                width = 300,
                key = 'time_series_year_slider'
            )

            # finally, filter to selected years
            time_series_year_mask = (
                (time_series_df["year"] >= time_series_year_range[0]) &
                (time_series_df["year"] <= time_series_year_range[1])
            )
            time_series_df = time_series_df.loc[time_series_year_mask]

            ########################################################################

            # define outcome breaks based on the country df 
            # - this is only used if the check box is ticked near the plot
            df_outcome_breaks = {}
            if selected_outcome_columns and outcome_1_column not in [None, '[placeholder]']:
                for i, outcome_column_i in enumerate(selected_outcome_columns, start=1):
                    outcome_column_as_list = [outcome_column_i]

                    # use polar to get min and max, then pass this mini df to get outcome breaks
                    min_max_df = polars_get_two_row_min_max_df(data_df_path, outcome_column_i)
                    df_outcome_breaks[i] = get_nice_breaks_from_list_of_df_columns(
                        min_max_df,
                        outcome_column_as_list,
                        min_value = None,
                        min_breaks=3, 
                        max_breaks=5, 
                        buffer=1.1
                    )

            ########################################################################
            for i, outcome_column_i in enumerate(selected_outcome_columns, start=1):
                outcome_container = st.container()
                with outcome_container:

                    st.subheader(outcome_1_label) 

                    # provide option to make use of fixed df_outcome_breaks for this country (calculated above):
                    use_same_y_axis_for_all_areas_in_country = st.checkbox(
                        "Use same Y-axis for all areas in country", 
                        value=False,
                        key = f"use_y_axis_{outcome_column_i}"
                    )

                    left_col, spacer_col, right_col = st.columns([7.5, 1, 7.5])    

                    with left_col:           

                        fig, axes_flat = plot_monthly_data_with_optional_year_overlay_or_second_outcome(
                            plot_df=time_series_df,
                            outcome_column=outcome_column_i,
                            time_axis_column='yearmon',
                            plot_monthly_data_overlay_years= False,
                            use_supplied_outcome_breaks = use_same_y_axis_for_all_areas_in_country,
                            supplied_outcome_breaks = df_outcome_breaks[i],
                            use_supplied_second_outcome_breaks = use_same_y_axis_for_all_areas_in_country,
                        )
                        ax = axes_flat[0]
                        
                        if outcome_1_label:
                            fig.text(0.075, 0.5, f"{outcome_1_label}", va='center', ha='center', rotation='vertical', fontsize=14)
                            ax.set_title(f"{outcome_1_label} for {st.session_state.selected_area}\n", fontsize=14)
                        else: 
                            fig.text(0.075, 0.5, f"{outcome_column_i.title()}", va='center', ha='center', rotation='vertical', fontsize=14)    
                            ax.set_title(f"{outcome_column_i.title()} for {st.session_state.selected_area}\n", fontsize=14)

                        st.pyplot(fig, width='content')

                        plot_file_name = f'{selected_country} {outcome_column_i} - full time series.png'
                        make_plot_download_button(fig, plot_file_name)

                    with spacer_col:
                        pass

                    with right_col:

                        fig, axes_flat = plot_monthly_data_with_optional_year_overlay_or_second_outcome(
                            plot_df=time_series_df,
                            outcome_column=outcome_column_i,
                            time_axis_column='yearmon',
                            plot_monthly_data_overlay_years= True,
                            use_supplied_outcome_breaks = use_same_y_axis_for_all_areas_in_country,
                            supplied_outcome_breaks = df_outcome_breaks[i],
                            use_supplied_second_outcome_breaks = use_same_y_axis_for_all_areas_in_country,
                        )
                        ax = axes_flat[0]
                        
                        if outcome_1_label:
                            fig.text(0.075, 0.5, f"{outcome_1_label}", va='center', ha='center', rotation='vertical', fontsize=14)
                            ax.set_title(f"{outcome_1_label} for {st.session_state.selected_area} - Overlaying Years\n", fontsize=14)
                        else: 
                            fig.text(0.075, 0.5, f"{outcome_column_i.title()}", va='center', ha='center', rotation='vertical', fontsize=14)    
                            ax.set_title(f"{outcome_column_i.title()} for {st.session_state.selected_area} - Overlaying Years\n", fontsize=14)

                        st.pyplot(fig, width='content')

                        plot_file_name = f'{selected_country} {outcome_column_i} - monthly data - overlay years.png'
                        make_plot_download_button(fig, plot_file_name)

                    # put in dividers (unless last i in loop)
                    if i != len(selected_outcome_columns):
                        st.divider()

        else:
            st.warning(f"No time series data available for this area: '{st.session_state.selected_area}'")
    else:
        # This message is shown if current_selected_area_for_plot is the placeholder
        st.info("Please select an area from the map or dropdown to see the plots.")


###########################################################################
# Seasonality Metric Plots

# FIXME (longer term)
# to simplify this section, the loop over outcomes has been removed - can be reinstated

with seasonality_metrics_tab:

    st.markdown(
        f"### Seasonality Plots and Metrics for for '{st.session_state.selected_area} - ({st.session_state.selected_admin1})'" 
        f"&emsp;<span style='color:#5DADE2;'>Scroll Down to View/Download the Metrics Data Frame</span>",
        unsafe_allow_html=True
    )

    st.subheader("Choose the seasonality window size of interest (in months)")
    window_size_options = [2, 3, 4, 5, 6]
    default_index = window_size_options.index(3)
    selected_window_size = st.selectbox(
        "Select the seasonality window size in months:", 
        window_size_options,
        index=default_index,
        key = 'seasonality_metric_window_size_selectbox'
    )
    window_size_list = [selected_window_size]


    if st.session_state.selected_area != "-- Select an area --":
        
            
        if outcome_1_column:

            seasonality_metrics_df_path = os.path.join(
                seasonality_metrics_folder,
                f"{country_prefix}_seasonality_metrics_df.csv"
            )
            year_summary_df_path = os.path.join(
                seasonality_metrics_folder,
                f"{country_prefix}_year_summary_df.csv"
            )
                
            seasonality_metrics_df = polars_load_df_and_filter_to_selected_admin1_and_area(
                df_path= seasonality_metrics_df_path,
                admin1_column = admin1_column,
                area_column = area_column,
                selected_admin1 = st.session_state.selected_admin1,
                selected_area = st.session_state.selected_area
            )

            year_summary_df = polars_load_df_and_filter_to_selected_admin1_and_area(
                df_path= year_summary_df_path,
                admin1_column = admin1_column,
                area_column = area_column,
                selected_admin1 = st.session_state.selected_admin1,
                selected_area = st.session_state.selected_area
            )

            # remove markham columns and filter to selected window size
            year_summary_mask = year_summary_df['window_size'] == selected_window_size
            year_summary_df = (
                year_summary_df.loc[
                    year_summary_mask, 
                    ['GID_0', area_column, 'year', 'window_size', 'max_window_percentage', 'max_start_month']
                ]
            )
            year_summary_df['outcome'] = outcome_1_column

            ###################################################################

            # build year slider based years in on the df:
            year_min = int(seasonality_metrics_df["year"].min())
            year_max = int(seasonality_metrics_df["year"].max())
            st.markdown("### Select Year Range for the Plots")
            seasonality_metrics_year_range = st.slider(
                "Select year range:",
                min_value=year_min,
                max_value=year_max,
                value=(year_min, year_max),
                width = 300,
                key = 'seasonality_metrics_year_slider'
            )

            # build masks based on the slider
            seasonality_metrics_year_mask = (
                (seasonality_metrics_df["year"] >= seasonality_metrics_year_range[0]) &
                (seasonality_metrics_df["year"] <= seasonality_metrics_year_range[1])
            )
            year_summary_year_mask = (
                (year_summary_df["year"] >= seasonality_metrics_year_range[0]) &
                (year_summary_df["year"] <= seasonality_metrics_year_range[1])
            )

            # finally, filter both dfs used in this section to selected years
            year_summary_df = year_summary_df.loc[year_summary_year_mask]
            seasonality_metrics_df = seasonality_metrics_df.loc[seasonality_metrics_year_mask]

            ###################################################################


            outcome_container = st.container()
            with outcome_container:
                st.subheader(outcome_1_label)
                left_col, spacer_col, right_col = st.columns([7.5, 1, 7.5])    

                with left_col:
            
                    # Seasonality Plot Left Hand Side  - percent of annual totals in specified window
                    fig, ax = plot_percentage_of_annual_totals_by_month_or_in_windows(
                        seasonality_metrics_df=seasonality_metrics_df,
                        overlay_years=True,
                        plot_percentages_in_window=True,
                        window_size= selected_window_size
                    )
                    st.pyplot(fig, width='content')
                    plot_file_name = f'{selected_country} {outcome_1_column} - percentage of annual total in {selected_window_size} month window.png'
                    make_plot_download_button(fig, plot_file_name)

                    st.divider()
                    string_dataframe = year_summary_df.astype(str)
                    st.markdown(f"### Seasonality Metrics Data Frame for {st.session_state.selected_area}")
                    st.dataframe(string_dataframe, hide_index=True, height = 1100)
                    st.markdown("") # spacer only
                    make_df_download_button(
                        year_summary_df,
                        filename_and_label_text=f'Seasonality Metrics for {outcome_1_column} in {selected_window_size} month windows'
                    )

                with spacer_col:
                    pass

                with right_col:

                    # Seasonality Plot Right Hand Side - histogram of consistency of peak of specified window size
                    fig, ax = plot_start_month_consistency_histogram_using_year_summary_df(
                        year_summary_df=year_summary_df,
                        window_size= selected_window_size
                    )
                    st.pyplot(fig, width='content')
                    plot_file_name = f'{selected_country} {outcome_1_column} - consistency of start month for {selected_window_size} month peak.png'
                    make_plot_download_button(fig, plot_file_name)    

            # put in dividers (unless last i in loop)
            if i != len(selected_outcome_columns):
                st.divider()

# end of Seasonality Metrics Expander block


###########################################################################
    
# Map of Areas identified by a specified seasonality criterion

# run as a fragment (can update maps in this fragment, without affecting the district-specific parts)
@st.fragment
def display_maps_of_areas_using_criterion():

    st.subheader(f"Map of Areas identifed by Seasonality Criteria for '{outcome_1_label}'")    
    # area_column_list = [area_column]

    # st.subheader("Choose a seasonality window size to calculate metrics for the whole country")
    threshold_window_size_options = [2, 3, 4, 5, 6]
    threshold_default_index = threshold_window_size_options.index(3)
    threshold_selected_window_size = st.selectbox(
        "Select the seasonality window size in months:", 
        threshold_window_size_options,
        index=threshold_default_index,
        key = 'threshold_window_size_selectbox'
    )
    threshold_window_size_list = [threshold_selected_window_size]    

    percentage_options = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
    percentage_default_index = percentage_options.index(60)
    threshold_percentage = st.selectbox(
        "Select the percentage to use for the threshold:", 
        percentage_options,
        index=percentage_default_index,
        key = 'threshold_percentage_selectbox'
    )
    
    # st.subheader("Choose the bin size for the percentage colour scale")
    threshold_bin_size_options = [5, 10, 20, 25]
    threshold_bin_default_index = threshold_bin_size_options.index(10)
    threshold_bin_size = st.selectbox(
        "Select the bin size for the percentage colour scale:", 
        threshold_bin_size_options,
        index=threshold_bin_default_index,
        key = 'threshold_bin_size_selectbox'
    )

    threshold_summary_df_path = os.path.join(
        seasonality_metrics_folder,
        f"{country_prefix}_threshold_summary_df.csv"
    )
    
    threshold_summary_df = (
        polars_load_threshold_summary_df_filtered_by_window_size_and_threshold_percentage(
            df_path = threshold_summary_df_path,
            window_size = threshold_selected_window_size,
            threshold_percentage = threshold_percentage
        )
    )

    # merge threshold summary with gdf
    gdf_for_mapped_output = gdf.copy()
    gdf_for_mapped_output = gdf_for_mapped_output.merge(
        threshold_summary_df, 
        left_on=area_column_list, 
        right_on=area_column_list, 
        how="left"
    )

    threshold_col1, threshold_spacer_col, threshold_col2 = st.columns([7, 3, 7])   

    with threshold_col1:

        # Percent of Years over Threshold
        title_text = f"Percent of Years (1995-2024) in which rainfall in {threshold_selected_window_size} months\nexceeds {threshold_percentage}% of annual rainfall"
        fig, ax = map_percent_or_number_of_years_above_threshold(
            gdf_for_mapped_output,
            column='percent_over_threshold',
            cmap_name ='OrRd',
            percent_or_number='percent',
            percent_bin_size=threshold_bin_size,
            title= title_text,
            colorbar_label="Percent of Years\n",
            figsize=(16, 9),
            ax=None,
            legend_shrink=0.8,
            edgecolor='black',
            linewidth=0.5
        )
        st.pyplot(fig, width='content')

        # make map available for download
        map_percent_file_name = f'{selected_country} map - percent of years {outcome_1_column} over {threshold_percentage} in {threshold_selected_window_size} months.png'
        make_plot_download_button(fig, map_percent_file_name)


    with threshold_spacer_col:
        pass
    
    with threshold_col2:
        title_text = f"Number of Years (1995-2024) in which rainfall in {threshold_selected_window_size} months\nexceeds {threshold_percentage}% of annual rainfall"

        fig, ax = map_percent_or_number_of_years_above_threshold(
            gdf_for_mapped_output,
            column='count_over_threshold',
            cmap_name ='Spectral_r',
            percent_or_number='number',
            percent_bin_size=None,
            title= title_text,
            colorbar_label="Number of Years\n",
            figsize=(16, 9),
            ax=None,
            legend_shrink=0.8,
            edgecolor='black',
            linewidth=0.5
        )
        st.pyplot(fig, width='content')

        # make map available for download
        map_number_file_name = f'map - number of years {outcome_1_column} over {threshold_percentage} in {threshold_selected_window_size} months.png'
        make_plot_download_button(fig, map_number_file_name)


with threshold_map_tab:
    display_maps_of_areas_using_criterion()



###########################################################################
# climatology

# if the country has bimodal areas, load list of the bimodal areas
bimodal_areas_path = os.path.join(climatology_data_folder, f"{country_prefix}_'{area_column}'_with_biannual_regime.csv")
if os.path.exists(bimodal_areas_path):
    bimodal_areas_df = load_and_return_df(bimodal_areas_path)

# if the selected area is listed in the bimodal areas dataframe, process further this section:

# load_yearly_water_season_df (decorate to avoid reloading)
full_anomaly_df_path = os.path.join(climatology_data_folder, f"{country_prefix}_anomaly_df.csv")
anomaly_df = polars_load_df_and_filter_to_selected_admin1_and_area(
    df_path = full_anomaly_df_path,
    admin1_column = admin1_column,
    area_column = area_column,
    selected_admin1 = st.session_state.selected_admin1,
    selected_area = st.session_state.selected_area
)

full_yearly_water_season_df_path = os.path.join(climatology_data_folder, f"{country_prefix}_yearly_water_season_df.csv")
yearly_water_season_df = polars_load_df_and_filter_to_selected_admin1_and_area(
    df_path = full_yearly_water_season_df_path,
    admin1_column = admin1_column,
    area_column = area_column,
    selected_admin1 = st.session_state.selected_admin1,
    selected_area = st.session_state.selected_area
)

###########################################################################

with climatology_tab:

    st.subheader(f"Climatology for for '{st.session_state.selected_area} - ({st.session_state.selected_admin1})'")

    if st.session_state.selected_area != "-- Select an area --":

        climatology_col1, climatology_spacer_col, climatology_col2 = st.columns([7.5, 1, 7.5])   

        with climatology_col1:

            if not anomaly_df.empty:
                
                st.subheader(f"Climatological Water Season for '{st.session_state.selected_area}'")          

                fig, ax1, ax2 = plot_climatological_water_season_for_selected_area(
                    anomaly_df = anomaly_df, 
                    area_column = area_column, 
                    selected_area = st.session_state.selected_area
                )
                st.pyplot(fig, width='content')
                plot_file_name = f'{selected_country} - {st.session_state.selected_area} - climatological water season.png'
                make_plot_download_button(fig, plot_file_name)  

            else:
                st.warning(f"No climatology data for '{st.session_state.selected_area}' - area may have bi-annual seasonality, or data may be missing")

        with climatology_spacer_col:
            pass

        with climatology_col2:

            if not yearly_water_season_df.empty:

                st.subheader(f"Yearly Water Season for '{st.session_state.selected_area}'")   

                fig, ax = plot_yearly_water_season_for_selected_area(
                    yearly_water_season_df = yearly_water_season_df, 
                    area_column = area_column, 
                    selected_area = st.session_state.selected_area
                )
                st.pyplot(fig, width='content')
                plot_file_name = f'{selected_country} - {st.session_state.selected_area} - yearly water season.png'
                make_plot_download_button(fig, plot_file_name)  

            else:
                st.warning(f"No yearly water season data for '{st.session_state.selected_area}' - area may have bi-annual seasonality, or data may be missing")


###########################################################################

with rainfall_onset_tab:

    if st.session_state.selected_area != "-- Select an area --":

     
        st.subheader("Choose climatological water season (30-years of data) or a specific year")
        water_year_range = list(range(1995, 2025))
        water_year_options = ["Climatological (1995-2024)"] + water_year_range
        selected_water_season = st.selectbox(
            "Select climatological water season or specific year:", 
            water_year_options,
            key = 'water_year_selectbox'
        )

        rainfall_onset_legend_interval = st.number_input(
            label = 'Adjust the interval for the colour legend:',
            min_value = 1.0,
            max_value = 100.0,
            value = 10.0,
            step = 5.0
        )

        rainfall_onset_col1, rainfall_onset_spacer_col, rainfall_onset_col2 = st.columns([7, 3, 7   ])   

        if selected_water_season == "Climatological (1995-2024)":
            water_season_df_path = os.path.join(climatology_data_folder, f'{country_prefix}_water_season_df.csv') 
            water_season_df = pd.read_csv(water_season_df_path, encoding='UTF-8')
            # don't fix colour scale for climatology (create these as None for use in function call below)
            min_onset = None
            max_cessation = None
        else:
            water_season_df_path = os.path.join(climatology_data_folder, f'{country_prefix}_yearly_water_season_df.csv') 
            water_season_df = pd.read_csv(water_season_df_path, encoding='UTF-8')
            # get min onset, max cessation for colour scale before filtering to year
            min_onset = water_season_df['onset_doy'].min()
            max_cessation = water_season_df['cessation_doy'].max() 
            # now filter to year
            mask = water_season_df['year']==selected_water_season
            water_season_df = water_season_df.loc[mask].copy()
             

        gdf_for_rainfall_onset = gdf.copy()
        gdf_for_rainfall_onset = gdf_for_rainfall_onset.merge(
            water_season_df, 
            left_on=area_column_list, 
            right_on=area_column_list, 
            how="left"
        )

        # merge in and flag bimodal areas 
        if os.path.exists(bimodal_areas_path):
            gdf_for_rainfall_onset = gdf_for_rainfall_onset.merge(
                bimodal_areas_df,
                left_on=area_column_list, 
                right_on=area_column_list, 
                how="left"
            )
            # flag existing areas as annual
            gdf_for_rainfall_onset['seasonality_regime'] = gdf_for_rainfall_onset['seasonality_regime'].mask(
                gdf_for_rainfall_onset['seasonality_regime'].isna(), 
                'annual'
            )
        else:
            # if no bimodal areas in country, all are annual so don't need conditional logic
            gdf_for_rainfall_onset['seasonality_regime'] = 'annual'        
        

        with rainfall_onset_col1:

            if not gdf_for_rainfall_onset.empty:
                
                st.subheader(f"Rainfall Onset Map for {selected_country}")
                st.subheader(f"{selected_water_season}")          

                fig, ax = map_onset_or_cessation_from_water_season_gdf(
                    gdf_for_rainfall_onset,
                    onset_or_cessation = 'onset',
                    legend_interval = rainfall_onset_legend_interval,
                    colour_scheme = 'Spectral',
                    min_onset = min_onset,
                    max_cessation = max_cessation,
                    additional_boundary_outline_gdf = gdf_for_rainfall_onset,
                    additional_boundary_color='black',
                    save_path = None
                )
                if rainfall_onset_legend_interval<5.00:
                    ax.tick_params(axis='y', labelsize=8)
                st.pyplot(fig, width='content')
                plot_file_name = f'{selected_country} - rainfall onset map - {selected_water_season}.png'
                make_plot_download_button(fig, plot_file_name)  

        with rainfall_onset_spacer_col:
            pass

        with rainfall_onset_col2:

            if not gdf_for_rainfall_onset.empty:

                st.subheader(f"Rainfall Cessation Map for {selected_country}")
                st.subheader(f"{selected_water_season}")          

                fig, ax = map_onset_or_cessation_from_water_season_gdf(
                    gdf_for_rainfall_onset,
                    onset_or_cessation = 'cessation',
                    legend_interval = rainfall_onset_legend_interval,
                    colour_scheme = 'Spectral',
                    min_onset = min_onset,
                    max_cessation = max_cessation,
                    additional_boundary_outline_gdf = gdf_for_rainfall_onset,
                    additional_boundary_color='black',
                    save_path = None
                )                
                st.pyplot(fig, width='content')
                if rainfall_onset_legend_interval<5.00:
                    ax.tick_params(axis='y', labelsize=8)
                plot_file_name = f'{selected_country} - rainfall cessation map - {selected_water_season}.png'
                make_plot_download_button(fig, plot_file_name)                  