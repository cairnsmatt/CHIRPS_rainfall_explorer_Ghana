
# %% imports
import numpy as np

# %%
# function definition

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


