# multiple_objects_same_timeframe

## DataFrame Combiner

The `combine_dataframes` function, located in `dataframe_combiner.py`, provides a flexible way to merge multiple time-series DataFrames into a single main DataFrame. It is particularly useful when dealing with datasets from different sources that need to be aligned to a common time index and potentially resampled.

### Key Features:

*   **Time Series Resampling and Interpolation**: Automatically handles time series data with different frequencies. It uses a time-weighted linear interpolation (`method='time'`) to fill missing values after aligning data to a common set of timestamps. Extrapolation beyond the original data range of each additional DataFrame is prevented (`limit_area='inside'`).
*   **Alignment to Main DataFrame's Time Index**: All additional DataFrames are processed and then precisely aligned to the `DatetimeIndex` of the `main_df`.
*   **Selective Column Joining**: Users can specify exactly which columns from each additional DataFrame they want to include in the final combined DataFrame.
*   **Custom Processing Support**: For more complex scenarios, a custom processing function can be applied to each additional DataFrame after resampling and reindexing but before it's joined to the main DataFrame.
*   **Error Handling**: Includes robust error checking for common issues like mismatched input lengths, missing configuration keys, incorrect data types, or missing time columns/data columns.

### Basic Usage Example:

```python
import pandas as pd
from dataframe_combiner import combine_dataframes

# 1. Prepare your DataFrames
# Main DataFrame with a DatetimeIndex
main_df = pd.DataFrame({
    'main_value': [100, 101, 102, 103, 104, 105]
}, index=pd.to_datetime([
    '2023-01-01 10:00:00', '2023-01-01 10:01:00', '2023-01-01 10:02:00',
    '2023-01-01 10:03:00', '2023-01-01 10:04:00', '2023-01-01 10:05:00'
]))

# Additional DataFrame with its own time column
additional_df1 = pd.DataFrame({
    'time': pd.to_datetime([
        '2023-01-01 10:00:30', '2023-01-01 10:02:30', '2023-01-01 10:04:30'
    ]),
    'sensor_value_A': [50.5, 52.0, 55.5],
    'sensor_value_B': [200, 205, 210]
})

# 2. Define Merge Configuration
# List of configurations, one for each additional DataFrame
merge_configs = [
    {
        'time_col': 'time',  # Name of the column in additional_df1 representing time
        'columns_to_join': ['sensor_value_A'], # List of columns to select and join
        # 'custom_processing_func': lambda df: df * 2 # Optional custom function
    }
]

# 3. Combine the DataFrames
combined_df = combine_dataframes(main_df, [additional_df1], merge_configs)

# 4. View the result
print(combined_df)

# Expected Output (values for sensor_value_A_add_0 will be interpolated):
#                      main_value  sensor_value_A_add_0
# 2023-01-01 10:00:00         100                  NaN  <-- or interpolated if data started earlier
# 2023-01-01 10:01:00         101                50.875
# 2023-01-01 10:02:00         102                51.625
# 2023-01-01 10:03:00         103                52.75
# 2023-01-01 10:04:00         104                54.75
# 2023-01-01 10:05:00         105                  NaN  <-- or interpolated if data ended later
# Note: Actual NaN or interpolated values at boundaries depend on the exact range of additional_df 
# relative to main_df and the interpolation method not extrapolating.
# The example values 50.875 etc. are illustrative of time-based interpolation.
```

The function ensures that `additional_df1`'s `sensor_value_A` is correctly interpolated and aligned with `main_df`'s timeline. If `sensor_value_B` was also needed, it would be added to the `columns_to_join` list. Column names from additional DataFrames are suffixed with `_add_N` (where N is the index of the additional DataFrame) to prevent naming collisions.
