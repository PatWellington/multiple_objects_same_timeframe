#!/usr/bin/env python3
import pandas as pd
import numpy as np

def combine_dataframes(main_df: pd.DataFrame, additional_dfs: list[pd.DataFrame], merge_configs: list[dict]) -> pd.DataFrame:
    """
    Combines additional DataFrames into a main DataFrame based on merge configurations.

    Args:
        main_df: The main DataFrame with a DatetimeIndex.
        additional_dfs: A list of additional DataFrames to merge.
        merge_configs: A list of dictionaries, each configuring how to merge an additional DataFrame.
                       Required keys: 'time_col', 'columns_to_join'.
                       Optional keys: 'custom_processing_func'.

    Returns:
        A single pandas DataFrame with data combined according to the configurations.

    Raises:
        ValueError: If inputs are invalid (e.g., main_df lacks DatetimeIndex,
                    mismatched lengths of additional_dfs and merge_configs,
                    missing required keys in configs).
    """
    if not isinstance(main_df.index, pd.DatetimeIndex):
        raise ValueError("Main DataFrame must have a DatetimeIndex.")

    if len(additional_dfs) != len(merge_configs):
        raise ValueError("Length of additional_dfs and merge_configs must be the same.")

    combined_df = main_df.copy()

    for i, (df_add, config) in enumerate(zip(additional_dfs, merge_configs)):
        if not isinstance(df_add, pd.DataFrame):
            raise ValueError(f"Item at index {i} in additional_dfs is not a DataFrame.")
        if not isinstance(config, dict):
            raise ValueError(f"Item at index {i} in merge_configs is not a dictionary.")

        time_col = config.get('time_col')
        columns_to_join = config.get('columns_to_join')
        custom_processing_func = config.get('custom_processing_func')

        if not time_col:
            raise ValueError(f"Missing 'time_col' in merge_configs at index {i}.")
        if not columns_to_join:
            raise ValueError(f"Missing 'columns_to_join' in merge_configs at index {i}.")
        if not isinstance(columns_to_join, list):
            raise ValueError(f"'columns_to_join' in merge_configs at index {i} must be a list.")
        if not all(isinstance(col, str) for col in columns_to_join):
            raise ValueError(f"All elements in 'columns_to_join' at index {i} must be strings.")


        if time_col not in df_add.columns:
            raise ValueError(f"'{time_col}' not found in additional DataFrame at index {i}.")
        if not pd.api.types.is_datetime64_any_dtype(df_add[time_col]):
             try:
                df_add[time_col] = pd.to_datetime(df_add[time_col])
             except Exception as e:
                raise ValueError(f"Could not convert '{time_col}' to datetime in additional DataFrame at index {i}: {e}")


        df_processed = df_add.set_index(time_col)

        if not isinstance(df_processed.index, pd.DatetimeIndex):
            # This should ideally not happen if previous checks pass, but as a safeguard:
            raise ValueError(f"Failed to set DatetimeIndex for additional DataFrame at index {i} using column '{time_col}'.")

        missing_cols = [col for col in columns_to_join if col not in df_processed.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in additional DataFrame at index {i}.")

        df_processed = df_processed[columns_to_join]

        # Ensure df_processed is sorted by index for reliable interpolation
        df_processed = df_processed.sort_index()

        # Create a combined index that includes all points from main_df's index (which is combined_df.index at this stage)
        # and df_processed's index. This ensures that when we interpolate, we do so over a grid
        # that respects all original data points from both sources.
        combined_union_index = combined_df.index.union(df_processed.index).sort_values()

        # Reindex df_processed to this combined_union_index. This aligns df_processed to a common grid
        # and introduces NaNs where main_df might have points outside df_processed's original range, and vice-versa.
        df_aligned_to_union = df_processed.reindex(combined_union_index)

        # Interpolate over this aligned data. This will fill NaNs based on all available data points
        # that are "inside" the valid data range. It will not extrapolate.
        # Using method='time' for time-weighted interpolation.
        df_aligned_to_union = df_aligned_to_union.infer_objects(copy=False)

        # Separate numerical and non-numerical columns
        numeric_cols = df_aligned_to_union.select_dtypes(include=np.number).columns
        non_numeric_cols = df_aligned_to_union.select_dtypes(exclude=np.number).columns

        df_numeric_interpolated = df_aligned_to_union[numeric_cols].interpolate(method='time', limit_area='inside')

        # For non-numeric columns, forward-fill (fill with the last valid observation) then back-fill (fill with next valid)
        # This is a common strategy for categorical/string data where interpolation is not meaningful.
        # Reindex to the union index first to ensure we cover all points, then fill.
        df_non_numeric_filled = df_aligned_to_union[non_numeric_cols].reindex(combined_union_index).ffill().bfill()

        # Combine interpolated numerical data with filled non-numerical data
        df_interpolated_on_union = pd.concat([df_numeric_interpolated, df_non_numeric_filled], axis=1)

        # Ensure the DataFrame is sorted by index if any operations above might have changed it
        df_interpolated_on_union = df_interpolated_on_union.sort_index()


        # Now, select only the timestamps that are in the main_df's current index.
        # This effectively downsamples or upsamples the interpolated data to match main_df's timeline,
        # and naturally handles cases where main_df's range is smaller or larger than df_processed.
        df_final_additional = df_interpolated_on_union.reindex(combined_df.index)

        if custom_processing_func:
            if not callable(custom_processing_func):
                raise ValueError(f"'custom_processing_func' in merge_configs at index {i} is not callable.")
            try:
                df_final_additional = custom_processing_func(df_final_additional)
            except Exception as e:
                raise ValueError(f"Error applying custom_processing_func for additional DataFrame at index {i}: {e}")

        # Add suffixes to avoid column name collisions, based on original column names
        # Suffix will be _add_i_colname
        rename_map = {col: f"{col}_add_{i}" for col in df_final_additional.columns}
        df_final_additional = df_final_additional.rename(columns=rename_map)

        combined_df = combined_df.join(df_final_additional, how='left')

    return combined_df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combine multiple CSV files based on timestamp.")
    parser.add_argument("main_csv_path", help="Path to the main CSV file.")
    parser.add_argument("additional_csv_paths", nargs='+', help="Paths to additional CSV files to combine.")

    args = parser.parse_args()

    # Read the main DataFrame
    main_df = pd.read_csv(args.main_csv_path)
    if 'timestamp' not in main_df.columns:
        raise ValueError("Main CSV must contain a 'timestamp' column.")
    try:
        main_df['timestamp'] = pd.to_datetime(main_df['timestamp'])
    except Exception as e:
        raise ValueError(f"Could not convert 'timestamp' column in main CSV to datetime: {e}")
    main_df = main_df.set_index('timestamp').sort_index()

    additional_dfs = []
    for path in args.additional_csv_paths:
        additional_dfs.append(pd.read_csv(path))

    # Define merge configurations for each additional DataFrame
    # This is a generic configuration, assuming the time column in additional CSVs is 'timestamp'
    # and we want to join all other columns.
    # Column names in additional CSVs are assumed to be unique or will be made unique by suffixing.
    merge_configs = []
    for i, df_add in enumerate(additional_dfs):
        if 'timestamp' not in df_add.columns:
            raise ValueError(f"Additional CSV at path {args.additional_csv_paths[i]} must contain a 'timestamp' column.")

        # Identify columns to join (all columns except 'timestamp')
        columns_to_join = [col for col in df_add.columns if col != 'timestamp']
        if not columns_to_join:
            raise ValueError(f"No columns to join in additional CSV {args.additional_csv_paths[i]} (excluding 'timestamp').")

        merge_configs.append({
            'time_col': 'timestamp',
            'columns_to_join': columns_to_join
        })

    combined_df = combine_dataframes(main_df, additional_dfs, merge_configs)

    # Print the combined DataFrame to stdout as CSV
    print(combined_df.to_csv())
