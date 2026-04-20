
import streamlit as st
import io

import numpy as np
import pandas as pd
import geopandas as gpd

import polars as pl

# %% --------------------------------------------------------------------------
def apply_decorator(decorator, apply_decorator=True):
    def decorate(func):
        if apply_decorator:
            return decorator(func)
        return func
    return decorate    
    # apply_decorator returns a function, decorate, 
    # which is used as in decorate(another_function)
    # decorated_function = apply_decorator(streamlit_decorator, apply_streamlit_decorator_flag)(undecorated_function)

# # # # example use
# # # streamlit_decorator = st.cache_data
# # # apply_streamlit_decorator_flag = True

# # # decorated_function = apply_decorator(streamlit_decorator, apply_streamlit_decorator_flag)(undecorated_function)
# # # my_function = apply_decorator(streamlit_decorator, apply_streamlit_decorator_flag)(my_function)


# %% --------------------------------------------------------------------------
# load and return pandas df as is
@st.cache_data
def load_and_return_df(df_path):
    df = pd.read_csv(df_path, encoding='UTF-8')
    return df


# %% --------------------------------------------------------------------------
# load monthly df, adding yearmon
@st.cache_data
def load_monthly_df_adding_yearmon(data_df_path, country_prefix):
    df = pd.read_csv(data_df_path)
    country_mask = df['GID_0'].isin([country_prefix])
    df = df.loc[country_mask].copy()
    # need to add yearmon for data columns
    df['yearmon'] = df['year'] + ((df['month'] - 1) / 12)
    df['yearmon'] = df['yearmon'].astype(np.float32)

    # if country_prefix == "TZA":
    #     area_column_original = f"{area_column}_original"
    #     df[area_column_original] = df[area_column]
    #     df[area_column] = df[area_column] + '_' + df['GID_2'].str.extract(r'TZA\.(.*)')[0]
    
    return df.copy()



# %% --------------------------------------------------------------------------
# load time series df for the selected district, adding yearmon
@st.cache_data
def polars_load_area_time_series_adding_yearmon(data_df_path, area_column, selected_area):

    query = (
        pl.scan_csv(data_df_path)
        .filter(
            (pl.col(area_column) == selected_area)
        )
    )
    # query.explain()
    df = query.collect()
    df = df.to_pandas()

    # add yearmon for data columns
    df['yearmon'] = df['year'] + ((df['month'] - 1) / 12)
    df['yearmon'] = df['yearmon'].astype(np.float32)

    return df.copy()


# %% --------------------------------------------------------------------------
# load time series df for a selected admin1 and area, adding yearmon
@st.cache_data
def polars_load_admin1_and_area_time_series_adding_yearmon(
    data_df_path, 
    admin1_column, 
    area_column, 
    selected_admin1,
    selected_area
):

    query = (
        pl.scan_csv(data_df_path)
        .filter(
            (pl.col(area_column) == selected_area) &
            (pl.col(admin1_column) == selected_admin1)
        )
    )
    # query.explain()
    df = query.collect()
    df = df.to_pandas()

    # add yearmon for data columns
    df['yearmon'] = df['year'] + ((df['month'] - 1) / 12)
    df['yearmon'] = df['yearmon'].astype(np.float32)

    return df.copy()


# %% --------------------------------------------------------------------------
# load df for a selected area
@st.cache_data
def polars_load_df_and_filter_to_selected_area(df_path, area_column, selected_area):

    query = (
        pl.scan_csv(df_path)
        .filter(
            (pl.col(area_column) == selected_area)
        )
    )
    df = query.collect()
    df = df.to_pandas()

    return df.copy()

# %% --------------------------------------------------------------------------
# load df for a selected admin1_column and area_column
@st.cache_data
def polars_load_df_and_filter_to_selected_admin1_and_area(
    df_path, 
    admin1_column, 
    area_column, 
    selected_admin1, 
    selected_area
):

    query = (
        pl.scan_csv(df_path)
        .filter(
            (pl.col(area_column) == selected_area) &
            (pl.col(admin1_column) == selected_admin1)
        )
    )
    df = query.collect()
    df = df.to_pandas()

    return df.copy()


# %% --------------------------------------------------------------------------
# use polars to get a two-row dataframe with the min and max for a particular outcome
@st.cache_data
def polars_get_two_row_min_max_df(df_path, outcome_column):
    
    query = (
        pl.scan_csv(df_path)
        .select([
            pl.col(outcome_column).min().alias("min"),
            pl.col(outcome_column).max().alias("max")
        ])
    )
    min_max_df = (
        query
        .collect()
        .melt(variable_name="value_type", value_name=outcome_column)
        .to_pandas()
    )

    return min_max_df


# %% --------------------------------------------------------------------------
# load threshold_summary_df filtered to window size and threshold percentage

@st.cache_data
def polars_load_threshold_summary_df_filtered_by_window_size_and_threshold_percentage(df_path, window_size, threshold_percentage):

    query = (
        pl.scan_csv(df_path)
        .filter(
            (pl.col('window_size') == window_size)
            &
            (pl.col('threshold_percentage') == threshold_percentage)
        )
    )
    df = query.collect()
    df = df.to_pandas()

    return df


# %% --------------------------------------------------------------------------
# create gdf from shapefile path
@st.cache_resource
def create_gdf_from_shapefile_path(country_shapefile_path, area_column):
    gdf = gpd.read_file(country_shapefile_path)

    # Simplify geometry for outline map
    gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.01, preserve_topology=True)

    if area_column not in gdf.columns: # Basic check for area_column
        st.error(f"Area column '{area_column}' not found in shapefile. Please check configuration.")
        st.stop()

    gdf_proj = gdf.to_crs(epsg=3857)   # web mercator
    centroids = gdf_proj.geometry.centroid
    centroids_latlon = centroids.to_crs(epsg=4326)
    mean_lat = centroids_latlon.y.mean()
    mean_lon = centroids_latlon.x.mean()    
    
    return gdf.copy(), gdf_proj, centroids, centroids_latlon, mean_lat, mean_lon


# %% --------------------------------------------------------------------------
@st.fragment
def make_plot_download_button(fig, plot_file_name):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches = 'tight')
    buf.seek(0)
    st.download_button(
        label = f"Download this plot as a PNG",
        data = buf,
        file_name = plot_file_name,
        mime = 'image/png'
    )

# %% --------------------------------------------------------------------------
@st.fragment
def make_df_download_button(
    df, 
    filename_and_label_text = 'this'
):
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index = False)
    csv_buffer.seek(0)
    st.download_button(
        label = f"Download {filename_and_label_text} as a CSV",
        data = csv_buffer,
        file_name = f"{filename_and_label_text} df.csv",
        mime = 'text/csv'
    )

