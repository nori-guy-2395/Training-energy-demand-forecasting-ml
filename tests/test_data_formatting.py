from energy_ml.data.loader import Data_formatting_clean_reassemble



import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "Date": ["01/01/2024", "02/01/2024", "bad_date"],
        "Time": ["12:00:00", "13:00:00", "bad_time"],
        "Power": ["1.5", "2.0", "?"]
    })


def test_returns_dataframe_and_issues(raw_df):
    clean_df, issues = Data_formatting_clean_reassemble(raw_df)

    assert isinstance(clean_df, pd.DataFrame)
    assert isinstance(issues, dict)
    
def test_date_parsing(raw_df):
    clean_df, issues = Data_formatting_clean_reassemble(raw_df)

    assert pd.api.types.is_datetime64_any_dtype(clean_df["Date"])
    assert clean_df["Date"].isna().sum() == 1  # bad_date
    
def test_time_parsing(raw_df):
    clean_df, issues = Data_formatting_clean_reassemble(raw_df)

    assert pd.api.types.is_datetime64_any_dtype(clean_df["Time"])
    assert clean_df["Time"].isna().sum() == 1  # bad_time    
    
def test_numeric_parsing(raw_df):
    clean_df, issues = Data_formatting_clean_reassemble(raw_df)

    assert pd.api.types.is_float_dtype(clean_df["Power"])
    assert clean_df["Power"].isna().sum() == 1  # "?"    
    
def test_datetime_column_created(raw_df):
    clean_df, _ = Data_formatting_clean_reassemble(raw_df)

    assert "Datetime" in clean_df.columns
    assert clean_df.columns[0] == "Datetime"    
    
def test_issue_tracking(raw_df):
    _, issues = Data_formatting_clean_reassemble(raw_df)

    assert "Date" in issues
    assert "Time" in issues
    assert "Power" in issues    
    
# def test_add_time_features(sample_df_with_datetime):
#     df_feat = add_time_features(sample_df_with_datetime)
#     for col in ["hour", "dayofweek", "month"]:
#         assert col in df_feat.columns
#         assert pd.api.types.is_integer_dtype(df_feat[col])