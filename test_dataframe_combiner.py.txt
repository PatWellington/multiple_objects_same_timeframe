import pandas as pd
import pytest
from dataframe_combiner import combine_dataframes
import numpy as np

# --- Helper Functions & Fixtures ---
@pytest.fixture
def sample_main_df():
    return pd.DataFrame({
        'value_main': range(10)
    }, index=pd.to_datetime([f'2023-01-01 00:0{i}:00' for i in range(10)])) # 1-minute frequency

@pytest.fixture
def sample_main_df_hourly():
    return pd.DataFrame({
        'value_main': range(3)
    }, index=pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 01:00:00', '2023-01-01 02:00:00']))


@pytest.fixture
def additional_df1():
    return pd.DataFrame({
        'time': pd.to_datetime([f'2023-01-01 00:0{i}:00' for i in range(5)]), # Same frequency, subset
        'sensor_a': [10, 11, 12, 13, 14],
        'sensor_b': [20, 21, 22, 23, 24]
    })

@pytest.fixture
def additional_df2_irregular_higher_freq():
    return pd.DataFrame({
        'timestamp': pd.to_datetime([
            '2023-01-01 00:00:00', '2023-01-01 00:00:30', '2023-01-01 00:01:00', # Covers first main_df point
            '2023-01-01 00:02:00', '2023-01-01 00:02:30', '2023-01-01 00:03:00'  # Covers third main_df point
        ]),
        'metric_x': [1.0, 1.5, 2.0, 3.0, 3.5, 4.0],
        'metric_y': [10.0, 10.5, 11.0, 12.0, 12.5, 13.0]
    })

@pytest.fixture
def additional_df3_lower_freq():
    return pd.DataFrame({
        'datetime_col': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:05:00', '2023-01-01 00:10:00']), # Lower frequency
        'feature_z': [100, 200, 300]
    })

def custom_func_multiply(df):
    return df * 2

# --- Test Cases ---

def test_basic_merge_compatible_time_series(sample_main_df, additional_df1):
    merge_config = [{
        'time_col': 'time',
        'columns_to_join': ['sensor_a']
    }]
    combined = combine_dataframes(sample_main_df, [additional_df1], merge_config)
    assert 'sensor_a_add_0' in combined.columns
    assert combined.loc['2023-01-01 00:00:00', 'sensor_a_add_0'] == 10
    assert combined.loc['2023-01-01 00:04:00', 'sensor_a_add_0'] == 14
    assert combined['sensor_a_add_0'].isna().sum() == 5 # Last 5 entries should be NaN as additional_df1 is shorter

def test_interpolation_higher_frequency_to_lower(sample_main_df, additional_df2_irregular_higher_freq):
    merge_config = [{
        'time_col': 'timestamp',
        'columns_to_join': ['metric_x']
    }]
    combined = combine_dataframes(sample_main_df, [additional_df2_irregular_higher_freq], merge_config)
    assert 'metric_x_add_0' in combined.columns
    # Check interpolated values
    # At 00:00:00, value is 1.0
    assert combined.loc['2023-01-01 00:00:00', 'metric_x_add_0'] == 1.0
    # At 00:01:00, value is 2.0
    assert combined.loc['2023-01-01 00:01:00', 'metric_x_add_0'] == 2.0
    # At 00:02:00, value is 3.0
    assert combined.loc['2023-01-01 00:02:00', 'metric_x_add_0'] == 3.0
     # At 00:03:00, value is 4.0
    assert combined.loc['2023-01-01 00:03:00', 'metric_x_add_0'] == 4.0

    # main_df has points at 00:00:00, 00:01:00, 00:02:00, 00:03:00, ...
    # additional_df2 has points at 00:00:00, 00:00:30, 00:01:00, 00:02:00, 00:02:30, 00:03:00
    # Expected:
    # 00:00:00 -> 1.0 (direct match)
    # 00:01:00 -> 2.0 (direct match)
    # 00:02:00 -> 3.0 (direct match)
    # 00:03:00 -> 4.0 (direct match)
    # Other points in main_df will be NaN or interpolated based on reindex behavior after resample.
    # The resampling happens TO main_df's frequency, then reindexed to main_df's index.
    # So, we should get values where main_df has indices.
    # For points in main_df not covered by additional_df2's range after resampling, they will be NaN.
    assert pd.isna(combined.loc['2023-01-01 00:04:00', 'metric_x_add_0']) # Outside range of additional_df2

def test_interpolation_lower_frequency_to_higher(sample_main_df, additional_df3_lower_freq):
    merge_config = [{
        'time_col': 'datetime_col',
        'columns_to_join': ['feature_z']
    }]
    combined = combine_dataframes(sample_main_df, [additional_df3_lower_freq], merge_config)
    assert 'feature_z_add_0' in combined.columns
    # Check interpolated values
    assert combined.loc['2023-01-01 00:00:00', 'feature_z_add_0'] == 100 # Direct match
    assert combined.loc['2023-01-01 00:01:00', 'feature_z_add_0'] == 100 + (200-100)/5 * 1 # Linear interpolation
    assert combined.loc['2023-01-01 00:02:00', 'feature_z_add_0'] == 100 + (200-100)/5 * 2
    assert combined.loc['2023-01-01 00:03:00', 'feature_z_add_0'] == 100 + (200-100)/5 * 3
    assert combined.loc['2023-01-01 00:04:00', 'feature_z_add_0'] == 100 + (200-100)/5 * 4
    assert combined.loc['2023-01-01 00:05:00', 'feature_z_add_0'] == 200 # Direct match
        # Value at 00:09:00 is interpolated between 200 (at 00:05) and 300 (at 00:10 from add_df3)
        # Expected: 200 + ( (300-200) / (10-5 minutes) ) * (9-5 minutes) = 200 + (100/5)*4 = 200 + 20*4 = 200 + 80 = 280
                                                                        # after resample and interpolate, the last value is 300 at 00:10:00.
                                                                        # Reindexing to main_df.index should keep 300 if main_df.index extends to 00:10:00 or beyond.
                                                                        # sample_main_df ends at 00:09:00.
                                                                        # So, the value at 00:09:00 should be interpolated between 200 (at 00:05:00) and 300 (at 00:10:00)
    # Value at 00:09:00: ( (300-200) / (10-5) ) * (9-5) + 200 = (100/5)*4 + 200 = 20*4+200 = 80+200 = 280
    assert combined.loc['2023-01-01 00:09:00', 'feature_z_add_0'] == 280.0


def test_time_alignment_subset(sample_main_df, additional_df1): # additional_df1 is a subset
    merge_config = [{'time_col': 'time', 'columns_to_join': ['sensor_a']}]
    combined = combine_dataframes(sample_main_df, [additional_df1], merge_config)
    assert not combined['sensor_a_add_0'].iloc[0:5].isna().any()
    assert combined['sensor_a_add_0'].iloc[5:].isna().all()

def test_time_alignment_superset(sample_main_df):
    df_super = pd.DataFrame({
            'time': pd.to_datetime([f'2023-01-01 00:{i:02d}:00' for i in range(12)]), # Superset of main_df
        'value': range(12)
    })
    merge_config = [{'time_col': 'time', 'columns_to_join': ['value']}]
    combined = combine_dataframes(sample_main_df, [df_super], merge_config)
    assert 'value_add_0' in combined.columns
    assert not combined['value_add_0'].isna().any() # All main_df times should be covered
    assert len(combined) == len(sample_main_df)

def test_time_alignment_partial_overlap(sample_main_df):
    df_overlap = pd.DataFrame({
        'time': pd.to_datetime([f'2023-01-01 00:0{i+3}:00' for i in range(5)]), # Overlaps from 00:03 to 00:07
        'value': range(5)
    })
    merge_config = [{'time_col': 'time', 'columns_to_join': ['value']}]
    combined = combine_dataframes(sample_main_df, [df_overlap], merge_config)
    assert combined['value_add_0'].iloc[0:3].isna().all()
    assert not combined['value_add_0'].iloc[3:8].isna().any()
    assert combined['value_add_0'].iloc[8:].isna().all()


def test_use_of_time_col(sample_main_df, additional_df1):
    # additional_df1 has 'time' column. This test ensures 'time_col' config is used.
    renamed_df1 = additional_df1.rename(columns={'time': 'timestamp_custom'})
    merge_config = [{'time_col': 'timestamp_custom', 'columns_to_join': ['sensor_a']}]
    combined = combine_dataframes(sample_main_df, [renamed_df1], merge_config)
    assert 'sensor_a_add_0' in combined.columns
    assert combined.loc['2023-01-01 00:00:00', 'sensor_a_add_0'] == 10

def test_selective_column_joining(sample_main_df, additional_df1):
    merge_config = [{'time_col': 'time', 'columns_to_join': ['sensor_a']}] # Only sensor_a
    combined = combine_dataframes(sample_main_df, [additional_df1], merge_config)
    assert 'sensor_a_add_0' in combined.columns
    assert 'sensor_b_add_0' not in combined.columns

    merge_config_b = [{'time_col': 'time', 'columns_to_join': ['sensor_b']}] # Only sensor_b
    combined_b = combine_dataframes(sample_main_df, [additional_df1], merge_config_b)
    assert 'sensor_b_add_0' in combined_b.columns
    assert 'sensor_a_add_0' not in combined_b.columns

    merge_config_both = [{'time_col': 'time', 'columns_to_join': ['sensor_a', 'sensor_b']}] # Both
    combined_both = combine_dataframes(sample_main_df, [additional_df1], merge_config_both)
    assert 'sensor_a_add_0' in combined_both.columns
    assert 'sensor_b_add_0' in combined_both.columns


def test_custom_processing_func(sample_main_df, additional_df1):
    merge_config = [{
        'time_col': 'time',
        'columns_to_join': ['sensor_a'],
        'custom_processing_func': custom_func_multiply
    }]
    combined = combine_dataframes(sample_main_df, [additional_df1], merge_config)
    assert combined.loc['2023-01-01 00:00:00', 'sensor_a_add_0'] == 10 * 2
    assert combined.loc['2023-01-01 00:04:00', 'sensor_a_add_0'] == 14 * 2

def test_multiple_additional_dfs(sample_main_df, additional_df1, additional_df3_lower_freq):
    renamed_df3 = additional_df3_lower_freq.rename(columns={'feature_z': 'another_feature'})
    merge_configs = [
        {'time_col': 'time', 'columns_to_join': ['sensor_a']},
        {'time_col': 'datetime_col', 'columns_to_join': ['another_feature'], 'custom_processing_func': custom_func_multiply}
    ]
    combined = combine_dataframes(sample_main_df, [additional_df1, renamed_df3], merge_configs)
    assert 'sensor_a_add_0' in combined.columns
    assert 'another_feature_add_1' in combined.columns # Note suffix _add_1
    assert combined.loc['2023-01-01 00:00:00', 'sensor_a_add_0'] == 10
    assert combined.loc['2023-01-01 00:00:00', 'another_feature_add_1'] == 100 * 2


# --- Error Handling Test Cases ---

def test_error_main_df_no_datetimeindex():
    main_no_dt = pd.DataFrame({'value': range(5)})
    with pytest.raises(ValueError, match="Main DataFrame must have a DatetimeIndex."):
        combine_dataframes(main_no_dt, [], [])

def test_error_mismatched_lengths_dfs_configs(sample_main_df, additional_df1):
    merge_configs = [] # Empty, but additional_dfs has one item
    with pytest.raises(ValueError, match="Length of additional_dfs and merge_configs must be the same."):
        combine_dataframes(sample_main_df, [additional_df1], merge_configs)

    with pytest.raises(ValueError, match="Length of additional_dfs and merge_configs must be the same."):
        combine_dataframes(sample_main_df, [], [{'time_col': 't', 'columns_to_join': ['c']}])


def test_error_missing_time_col_in_config(sample_main_df, additional_df1):
    merge_config = [{'columns_to_join': ['sensor_a']}] # Missing 'time_col'
    with pytest.raises(ValueError, match="Missing 'time_col' in merge_configs at index 0."):
        combine_dataframes(sample_main_df, [additional_df1], merge_config)

def test_error_missing_columns_to_join_in_config(sample_main_df, additional_df1):
    merge_config = [{'time_col': 'time'}] # Missing 'columns_to_join'
    with pytest.raises(ValueError, match="Missing 'columns_to_join' in merge_configs at index 0."):
        combine_dataframes(sample_main_df, [additional_df1], merge_config)

def test_error_time_col_not_in_additional_df(sample_main_df, additional_df1):
    merge_config = [{'time_col': 'non_existent_time_col', 'columns_to_join': ['sensor_a']}]
    with pytest.raises(ValueError, match="'non_existent_time_col' not found in additional DataFrame at index 0."):
        combine_dataframes(sample_main_df, [additional_df1], merge_config)

def test_error_column_to_join_not_in_additional_df(sample_main_df, additional_df1):
    merge_config = [{'time_col': 'time', 'columns_to_join': ['non_existent_sensor']}]
    with pytest.raises(ValueError, match="Columns \\['non_existent_sensor'\\] not found in additional DataFrame at index 0."):
        combine_dataframes(sample_main_df, [additional_df1], merge_config)

def test_error_invalid_custom_processing_func(sample_main_df, additional_df1):
    merge_config = [{
        'time_col': 'time',
        'columns_to_join': ['sensor_a'],
        'custom_processing_func': "not_a_function"
    }]
    with pytest.raises(ValueError, match="'custom_processing_func' in merge_configs at index 0 is not callable."):
        combine_dataframes(sample_main_df, [additional_df1], merge_config)

def test_error_custom_processing_func_fails(sample_main_df, additional_df1):
    def failing_func(df):
        raise RuntimeError("Processing failed")
    merge_config = [{
        'time_col': 'time',
        'columns_to_join': ['sensor_a'],
        'custom_processing_func': failing_func
    }]
    with pytest.raises(ValueError, match="Error applying custom_processing_func for additional DataFrame at index 0: Processing failed"):
        combine_dataframes(sample_main_df, [additional_df1], merge_config)

def test_error_additional_df_item_not_dataframe(sample_main_df):
    merge_config = [{'time_col': 'time', 'columns_to_join': ['sensor_a']}]
    with pytest.raises(ValueError, match="Item at index 0 in additional_dfs is not a DataFrame."):
        combine_dataframes(sample_main_df, ["not_a_dataframe"], merge_config)

def test_error_merge_config_item_not_dict(sample_main_df, additional_df1):
    with pytest.raises(ValueError, match="Item at index 0 in merge_configs is not a dictionary."):
        combine_dataframes(sample_main_df, [additional_df1], ["not_a_dict"])

def test_error_columns_to_join_not_list(sample_main_df, additional_df1):
    merge_config = [{'time_col': 'time', 'columns_to_join': "sensor_a"}] # should be a list
    with pytest.raises(ValueError, match="'columns_to_join' in merge_configs at index 0 must be a list."):
        combine_dataframes(sample_main_df, [additional_df1], merge_config)
        
def test_error_columns_to_join_list_not_all_strings(sample_main_df, additional_df1):
    merge_config = [{'time_col': 'time', 'columns_to_join': ['sensor_a', 123]}] # 123 is not a string
    with pytest.raises(ValueError, match="All elements in 'columns_to_join' at index 0 must be strings."):
        combine_dataframes(sample_main_df, [additional_df1], merge_config)

def test_time_col_needs_conversion_to_datetime(sample_main_df):
    df_add = pd.DataFrame({
        'event_time': ['2023-01-01 00:00:00', '2023-01-01 00:01:00'],
        'value': [10, 20]
    }) # 'event_time' is string
    merge_config = [{'time_col': 'event_time', 'columns_to_join': ['value']}]
    combined = combine_dataframes(sample_main_df, [df_add], merge_config)
    assert 'value_add_0' in combined.columns
    assert combined.loc['2023-01-01 00:00:00', 'value_add_0'] == 10.0

def test_error_time_col_cannot_be_converted_to_datetime(sample_main_df):
    df_add = pd.DataFrame({
        'event_time': ['this is not a time', 'neither is this'],
        'value': [10, 20]
    })
    merge_config = [{'time_col': 'event_time', 'columns_to_join': ['value']}]
    with pytest.raises(ValueError, match="Could not convert 'event_time' to datetime in additional DataFrame at index 0"):
        combine_dataframes(sample_main_df, [df_add], merge_config)

def test_main_df_irregular_index(additional_df1):
    main_df_irregular = pd.DataFrame({
        'value_main': [1, 2, 3, 4]
    }, index=pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:00:30', '2023-01-01 00:01:30', '2023-01-01 00:03:00']))
    
    merge_config = [{
        'time_col': 'time', # additional_df1 is 00:00, 00:01, 00:02, 00:03, 00:04
        'columns_to_join': ['sensor_a']
    }]
    combined = combine_dataframes(main_df_irregular, [additional_df1], merge_config)
    assert 'sensor_a_add_0' in combined.columns
    pd.testing.assert_series_equal(
        combined['sensor_a_add_0'],
        pd.Series([10.0, 10.5, 11.5, 13.0], index=main_df_irregular.index, name='sensor_a_add_0'),
        check_dtype=False # allow float differences
    )

def test_main_df_freq_hourly_additional_minutely(sample_main_df_hourly, additional_df1):
    # main_df is hourly, additional_df1 is minutely from 00:00 to 00:04
    merge_config = [{
        'time_col': 'time',
        'columns_to_join': ['sensor_a']
    }]
    combined = combine_dataframes(sample_main_df_hourly, [additional_df1], merge_config)
    # Expected: additional_df1 will be resampled to hourly.
    # For '2023-01-01 00:00:00', sensor_a should be 10 (exact match)
    # For '2023-01-01 01:00:00', sensor_a should be NaN as additional_df1 doesn't cover this,
    # and linear interpolation of additional_df1 resampled to hourly would make it NaN beyond its original range.
    assert combined.loc['2023-01-01 00:00:00', 'sensor_a_add_0'] == 10.0
    assert pd.isna(combined.loc['2023-01-01 01:00:00', 'sensor_a_add_0'])
    assert pd.isna(combined.loc['2023-01-01 02:00:00', 'sensor_a_add_0'])
    assert len(combined) == 3

def test_empty_additional_dfs_and_configs(sample_main_df):
    combined = combine_dataframes(sample_main_df, [], [])
    assert combined.equals(sample_main_df) # Should return a copy of main_df

def test_additional_df_all_nans_after_reindex(sample_main_df):
    # Additional DF whose time range is completely outside main_df
    df_outside = pd.DataFrame({
        'time': pd.to_datetime(['2024-01-01 00:00:00', '2024-01-01 00:01:00']),
        'value': [100, 200]
    })
    merge_config = [{'time_col': 'time', 'columns_to_join': ['value']}]
    combined = combine_dataframes(sample_main_df, [df_outside], merge_config)
    assert 'value_add_0' in combined.columns
    assert combined['value_add_0'].isna().all()

def test_additional_df_some_nans_in_join_columns(sample_main_df):
    df_with_nans = pd.DataFrame({
        'time': pd.to_datetime([f'2023-01-01 00:00:00', f'2023-01-01 00:01:00', f'2023-01-01 00:02:00']),
        'sensor_x': [1.0, np.nan, 3.0]
    })
    merge_config = [{'time_col': 'time', 'columns_to_join': ['sensor_x']}]
    combined = combine_dataframes(sample_main_df, [df_with_nans], merge_config)
    assert combined.loc['2023-01-01 00:00:00', 'sensor_x_add_0'] == 1.0
    # Original test assertion was pd.isna(...), which is incorrect for linear interpolation.
    # Linear interpolation of [1.0, nan, 3.0] should yield [1.0, 2.0, 3.0]
    assert combined.loc['2023-01-01 00:01:00', 'sensor_x_add_0'] == 2.0
    assert combined.loc['2023-01-01 00:02:00', 'sensor_x_add_0'] == 3.0
    # The comment below is correct and confirms the value should be 2.0.
    # ( (3.0 - 1.0) / (pd.Timestamp('2023-01-01 00:02:00') - pd.Timestamp('2023-01-01 00:00:00')).total_seconds() * \
    #   (pd.Timestamp('2023-01-01 00:01:00') - pd.Timestamp('2023-01-01 00:00:00')).total_seconds() ) + 1.0
    # = (2.0 / 120 seconds) * 60 seconds + 1.0 = 1.0 + 1.0 = 2.0
    # The existing final assert already checks this.
    assert combined.loc['2023-01-01 00:01:00', 'sensor_x_add_0'] == 2.0

# Test to ensure original main_df is not modified
def test_main_df_not_modified(sample_main_df, additional_df1):
    main_df_copy = sample_main_df.copy()
    merge_config = [{'time_col': 'time', 'columns_to_join': ['sensor_a']}]
    combine_dataframes(sample_main_df, [additional_df1], merge_config)
    pd.testing.assert_frame_equal(sample_main_df, main_df_copy)


# Test to ensure additional_dfs are not modified
def test_additional_dfs_not_modified(sample_main_df, additional_df1):
    additional_df1_copy = additional_df1.copy()
    merge_config = [{'time_col': 'time', 'columns_to_join': ['sensor_a']}]
    combine_dataframes(sample_main_df, [additional_df1], merge_config)
    pd.testing.assert_frame_equal(additional_df1, additional_df1_copy)

# Test with main_df having no frequency (irregular index)
def test_main_df_irregular_index_no_freq_attr(additional_df1):
    # Create main_df with an irregular DatetimeIndex that won't have a .freq attribute
    main_df_irregular_no_freq = pd.DataFrame({
        'value_main': [100, 200, 300, 400]
    }, index=pd.to_datetime(['2023-01-01 00:00:15', '2023-01-01 00:00:45', '2023-01-01 00:01:35', '2023-01-01 00:02:55']))
    assert main_df_irregular_no_freq.index.freq is None

    merge_config = [{
        'time_col': 'time', # additional_df1 is minutely: 00:00, 00:01, 00:02, 00:03, 00:04
        'columns_to_join': ['sensor_a'] # values: 10, 11, 12, 13, 14
    }]
    
    combined = combine_dataframes(main_df_irregular_no_freq, [additional_df1], merge_config)
    
    # Expected values for sensor_a_add_0 at main_df_irregular_no_freq indices:
    # '2023-01-01 00:00:15': interpolated between 10 (at 00:00) and 11 (at 00:01). (11-10)/60 * 15 + 10 = 1/4 + 10 = 10.25
    # '2023-01-01 00:00:45': interpolated between 10 (at 00:00) and 11 (at 00:01). (11-10)/60 * 45 + 10 = 3/4 + 10 = 10.75
    # '2023-01-01 00:01:35': interpolated between 11 (at 00:01) and 12 (at 00:02). (12-11)/60 * 35 + 11 = 35/60 + 11 = 7/12 + 11 = 11.58333...
    # '2023-01-01 00:02:55': interpolated between 12 (at 00:02) and 13 (at 00:03). (13-12)/60 * 55 + 12 = 55/60 + 12 = 11/12 + 12 = 12.91666...
    
    expected_values = pd.Series(
        [10.25, 10.75, 11 + (35/60.0) , 12 + (55/60.0) ],
        index=main_df_irregular_no_freq.index,
        name='sensor_a_add_0'
    )
    pd.testing.assert_series_equal(combined['sensor_a_add_0'], expected_values, rtol=1e-5)

# Test case for when additional_df's time column is already the index
def test_additional_df_time_col_is_index(sample_main_df, additional_df1):
    additional_df1_indexed = additional_df1.set_index('time')
    # The function expects 'time_col' to be a column, so we need to reset index or adapt the function.
    # For now, let's assume the user might pass it this way and the function should handle it if 'time_col' is the index name.
    # Current implementation: df_add.set_index(time_col) will work fine if time_col is index name.
    # However, if time_col is not in df_add.columns, it will fail: if time_col not in df_add.columns
    # Let's test the case where time_col is the name of the index.
    
    # To make this testable with current code, we must ensure 'time_col' exists as a column.
    # So, if the index is already the time column, we might need to reset it.
    # OR, the function could be smarter. Let's test current behavior.
    
    # If 'time' is the index name:
    additional_df1_indexed.index.name = 'time_idx'
    df_to_pass = additional_df1_indexed.reset_index() # Make 'time_idx' a column

    merge_config = [{
        'time_col': 'time_idx', # Use the actual column name now
        'columns_to_join': ['sensor_a']
    }]
    combined = combine_dataframes(sample_main_df, [df_to_pass], merge_config)
    assert 'sensor_a_add_0' in combined.columns
    assert combined.loc['2023-01-01 00:00:00', 'sensor_a_add_0'] == 10

    # What if time_col is *already* the index and is named?
    # The line `if time_col not in df_add.columns:` will fail.
    # Let's try to make a df where time_col is the index name.
    df_indexed_named = additional_df1.set_index('time') # index is now DatetimeIndex named 'time'
    
    with pytest.raises(ValueError, match="'time' not found in additional DataFrame at index 0"):
        combine_dataframes(sample_main_df, [df_indexed_named], [{'time_col': 'time', 'columns_to_join': ['sensor_a']}])

    # If the function were to be enhanced, it might check:
    # if time_col == df_add.index.name: df_processed = df_add.copy()
    # else: df_processed = df_add.set_index(time_col) etc.
    # But for now, the current implementation requires time_col to be an actual column.
    # So the test above with `df_to_pass` is the valid way to use the current function.

# Test suffix naming convention for multiple columns from the same additional DataFrame
def test_suffix_naming_multiple_columns(sample_main_df, additional_df1):
    merge_config = [{
        'time_col': 'time',
        'columns_to_join': ['sensor_a', 'sensor_b']
    }]
    combined = combine_dataframes(sample_main_df, [additional_df1], merge_config)
    assert 'sensor_a_add_0' in combined.columns
    assert 'sensor_b_add_0' in combined.columns
    assert combined.loc['2023-01-01 00:00:00', 'sensor_a_add_0'] == 10
    assert combined.loc['2023-01-01 00:00:00', 'sensor_b_add_0'] == 20
