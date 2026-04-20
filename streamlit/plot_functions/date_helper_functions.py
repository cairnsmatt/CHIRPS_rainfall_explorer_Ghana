# %% --------------------------------------------------------------------------
import calendar

# %% --------------------------------------------------------------------------
# date helper functions:

def convert_year_and_month_into_time_column_value(year, month, origin_year, origin_month):

    time_column_value = (year - origin_year) * 12 + (month - origin_month)
    
    return time_column_value


# helper function to determine number of weeks in year
def iso_weeks_in_year(year):
    jan1_weekday = calendar.weekday(year, 1, 1)
    is_leap = calendar.isleap(year)

    if jan1_weekday == 3 or (jan1_weekday == 2 and is_leap):
        return 53
    return 52


# helper to convert iso year and iso week into continuous weekly time_column_value
def convert_year_and_week_into_time_column_value(
    year,
    week,
    origin_year,
    origin_week=1
):

    # Calculate cumulative week offset between origin_year and target year
    if year < origin_year:
        raise ValueError("year must be >= origin_year")

    weeks_offset = 0

    for y in range(origin_year, year):
        weeks_offset += iso_weeks_in_year(y)

    time_column_value = weeks_offset + (week - origin_week)

    return time_column_value
