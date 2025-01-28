import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.preprocessing import KBinsDiscretizer

# Setup logger
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger

logger = setup_logger("rfms_woe_binning")


def validate_datetime_column(column, timezone="Africa/Addis_Ababa"):
    
    if not pd.api.types.is_datetime64_any_dtype(column):
        column = pd.to_datetime(column, errors='coerce')
    column = (
        column.dt.tz_localize(None)#.dt.tz_localize(pytz.UTC).dt.tz_convert(timezone) 
        if column.dt.tz is not None
        else column
    )
    return column
    

def calculate_rfms(data: pd.DataFrame, customer_id_col: str, transaction_date_col: str, amount_col: str) -> pd.DataFrame:
    """
    Calculate RFMS (Recency, Frequency, Monetary, Stability) scores for each customer.
    """
    # Convert transaction date to datetime
    data[transaction_date_col] = validate_datetime_column(data[transaction_date_col])

    # Calculate Recency (days since last transaction)
    recency = data.groupby(customer_id_col)[transaction_date_col].max().reset_index()
    recency['Recency'] = (pd.to_datetime('today') - recency[transaction_date_col]).dt.days

    # Calculate Frequency (number of transactions)
    frequency = data.groupby(customer_id_col).size().reset_index(name='Frequency')

    # Calculate Monetary (total transaction amount)
    monetary = data.groupby(customer_id_col)[amount_col].sum().reset_index(name='Monetary')

    # Calculate Stability (standard deviation of transaction amounts)
    stability = data.groupby(customer_id_col)[amount_col].std().reset_index(name='Stability')
    stability['Stability'] = stability['Stability'].fillna(0) # for single or no transactions 

    # Merge RFMS metrics
    rfms = recency.merge(frequency, on=customer_id_col) \
                  .merge(monetary, on=customer_id_col) \
                  .merge(stability, on=customer_id_col)

    return rfms

def assign_good_bad_labels(rfms: pd.DataFrame, recency_threshold: int, frequency_threshold: int, 
                           monetary_threshold: float, stability_threshold: float) -> pd.DataFrame:
    """
    Assign Good/Bad labels based on RFMS scores.
    """
    rfms['RFMS_Label'] = np.where(
        (rfms['Recency'] <= recency_threshold) &
        (rfms['Frequency'] >= frequency_threshold) &
        (rfms['Monetary'] >= monetary_threshold) &
        (rfms['Stability'] <= stability_threshold),
        'Good', 'Bad'
    )
    return rfms


def calculate_woe_iv(data: pd.DataFrame, feature: str, target: str, n_bins: int = 10) -> Tuple[pd.DataFrame, float]:
    """
    Calculate Weight of Evidence (WoE) and Information Value (IV) for a feature.
    """
    # Discretize the feature into bins
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    data[f'{feature}_binned'] = discretizer.fit_transform(data[[feature]])

    # Calculate WoE and IV
    woe_table = data.groupby(f'{feature}_binned').agg(
        Total=pd.NamedAgg(column=target, aggfunc='count'),
        Good=pd.NamedAgg(column=target, aggfunc=lambda x: (x == 'Good').sum()),
        Bad=pd.NamedAgg(column=target, aggfunc=lambda x: (x == 'Bad').sum())
    ).reset_index()

    woe_table['Good%'] = woe_table['Good'] / (woe_table['Good'].sum() or 1)
    woe_table['Bad%'] = woe_table['Bad'] / (woe_table['Bad'].sum() or 1)
    woe_table['WoE'] = np.log((woe_table['Good%'] + 1e-9) / (woe_table['Bad%'] + 1e-9))
    woe_table['IV'] = (woe_table['Good%'] - woe_table['Bad%']) * woe_table['WoE']

    iv_value = woe_table['IV'].sum()

    return woe_table, iv_value

def woe_pipeline(data: pd.DataFrame, customer_id_col: str, transaction_date_col: str, amount_col: str, 
                recency_threshold: int, frequency_threshold: int, monetary_threshold: float, 
                stability_threshold: float) -> Dict[str, pd.DataFrame]:
    """ Main workflow for Default Estimator and WoE Binning. """
    # Calculate RFMS scores
    rfms = calculate_rfms(data, customer_id_col, transaction_date_col, amount_col)

    # Assign Good/Bad labels
    assigned_rfms = assign_good_bad_labels(rfms, recency_threshold, frequency_threshold, monetary_threshold, stability_threshold)
    
    # Calculate WoE and IV for each RFMS feature
    woe_results = {}
    for feature in ['Recency', 'Frequency', 'Monetary', 'Stability']:
        woe_table, iv_value = calculate_woe_iv(assigned_rfms, feature, 'RFMS_Label')
        woe_results[feature] = {'WoE_Table': woe_table, 'IV': iv_value}

    return {'RFMS': assigned_rfms, 'WoE_Results': woe_results}