import os
import sys
import numpy as np
import pandas as pd
from xverse.transformer import WOE
from sklearn.tree import DecisionTreeClassifier

# Setup logger
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger

logger = setup_logger("woe_binning")

def woe_encode(data: pd.DataFrame, target: str, categorical_columns: list) -> pd.DataFrame:
    """
    Applies Weight of Evidence (WoE) encoding to categorical columns.

    Args:
        data (pd.DataFrame): The input DataFrame.
        target (str): Target variable.
        categorical_columns (list): List of categorical columns to encode.

    Returns:
        pd.DataFrame: DataFrame with WoE encoded features.
    """
    woe_encoder = WOE()
    data = data.copy()
    woe_data = woe_encoder.fit_transform(data[categorical_columns], data[target])
    data = pd.concat([data, woe_data], axis=1)
    logger.info("WoE encoding applied.")
    return data

def save_to_file(data: pd.DataFrame, output_path: str) -> None:
    """
    Saves the DataFrame to a file.

    Args:
        data (pd.DataFrame): The DataFrame to save.
        output_path (str): File path to save the DataFrame.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    logger.info(f"Data saved to {output_path}")


def calculate_woe_iv(data: pd.DataFrame, feature: str, target: str, output_path: str = None) -> pd.DataFrame:
    """
    Calculates WoE and IV for a given feature (categorical or binned numerical).
    Optionally saves the results to a file.

    Args:
        data (pd.DataFrame): Input DataFrame.
        feature (str): Feature name.
        target (str): Target column name.
        output_path (str): Optional file path to save the WoE/IV calculation.

    Returns:
        pd.DataFrame: DataFrame with WoE, IV, and statistics for bins/categories.
    """
    logger.info(f"Calculating WoE and IV for feature: {feature}")

    unique_values = data[feature].unique()
    stats = []

    for value in unique_values:
        total = len(data[data[feature] == value])
        good = len(data[(data[feature] == value) & (data[target] == 0)])
        bad = len(data[(data[feature] == value) & (data[target] == 1)])
        
        good_prop = good / (len(data[data[target] == 0]) + 1e-9)
        bad_prop = bad / (len(data[data[target] == 1]) + 1e-9)
        woe = np.log((good_prop + 1e-9) / (bad_prop + 1e-9))
        iv = (good_prop - bad_prop) * woe
        
        stats.append((value, total, good, bad, woe, iv))

    result_data = pd.DataFrame(stats, columns=['Value', 'Total', 'Good', 'Bad', 'WoE', 'IV'])
    result_data['Feature'] = feature
    result_data['IV_Sum'] = result_data['IV'].sum()
    
    if output_path:
        save_to_file(result_data, output_path)

    logger.info(f"Completed WoE and IV for feature: {feature}")
    return result_data


def apply_woe_encoding(data: pd.DataFrame, woe_data: pd.DataFrame, feature: str, output_path: str = None) -> pd.DataFrame:
    """
    Applies WoE encoding for a feature and optionally saves the output.

    Args:
        data (pd.DataFrame): Input DataFrame.
        woe_data (pd.DataFrame): DataFrame containing WoE values for bins/categories.
        feature (str): Feature column name.
        output_path (str): Optional file path to save the DataFrame.

    Returns:
        pd.DataFrame: DataFrame with WoE-encoded feature.
    """
    logger.info(f"Applying WoE encoding for feature: {feature}")
    
    # Map WoE values to the original bins
    woe_map = woe_data.set_index('Value')['WoE'].to_dict()
    # woe_map = woe_data.set_index('Bin')['WoE'].to_dict()
    data[f'{feature}_WoE'] = data[feature].map(woe_map)

    if output_path:
        save_to_file(data, output_path)
    
    logger.info(f"Applying WoE encoding for feature: {feature}")
    return data

def bin_numerical_feature(data: pd.DataFrame, feature: str, target: str, max_bins: int = 5, output_path: str = None) -> pd.DataFrame:
    """
    Bins a numerical feature using supervised binning, calculates WoE, and optionally saves the results.

    Args:
        data (pd.DataFrame): Input DataFrame.
        feature (str): Numerical feature to bin.
        target (str): Target column name.
        max_bins (int): Maximum number of bins.
        output_path (str): Optional file path to save the resulting bins and WoE/IV mappings.

    Returns:
        pd.DataFrame: DataFrame with bins and calculated WoE/IV.
    """
    logger.info(f"Binning numerical feature: {feature} with max bins: {max_bins}")

    dt = DecisionTreeClassifier(max_leaf_nodes=max_bins)
    dt.fit(data[[feature]], data[target])
    data[f'{feature}_Binned'] = dt.apply(data[[feature]])

    woe_data = calculate_woe_iv(data, feature=f'{feature}_Binned', target=target, output_path=output_path)
    logger.info(f"Completed binning and WoE calculation for feature: {feature}")
    return woe_data

def woe_encode_categorical(data: pd.DataFrame, categorical_features: list, target: str, output_dir: str = None) -> pd.DataFrame:
    """
    Applies WoE encoding to multiple categorical features and optionally saves results.

    Args:
        data (pd.DataFrame): Input DataFrame.
        categorical_features (list): List of categorical features.
        target (str): Target column.
        output_dir (str): Optional directory path to save WoE/IV mappings and encoded DataFrame.

    Returns:
        pd.DataFrame: DataFrame with WoE-encoded features.
    """
    logger.info("Starting WoE encoding for categorical features.")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for feature in categorical_features:
        output_path = os.path.join(output_dir, f'{feature}_woe.csv') if output_dir else None
        woe_data = calculate_woe_iv(data, feature, target, output_path)
        encoded_output = os.path.join(output_dir, f'{feature}_encoded.csv') if output_dir else None
        data = apply_woe_encoding(data, woe_data, feature, encoded_output)
        logger.info(f"WoE encoding completed for feature: {feature}")

    return data


def calculate_default_rate(data: pd.DataFrame, target: str, output_path: str = None) -> float:
    """
    Calculates the default rate for a binary target column and optionally saves it.

    Args:
        data (pd.DataFrame): Input DataFrame.
        target (str): Target column.
        output_path (str): Optional path to save the calculated default rate.

    Returns:
        float: Default rate.
    """
    default_rate = data[target].mean()
    if output_path:
        with open(output_path, 'w') as f:
            f.write(f'Default Rate: {default_rate}\n')
        logger.info(f"Default rate saved to {output_path}")
    
    logger.info(f"Default rate for target ({target}): {default_rate}")
    return default_rate

# def iv_woe(data, target, bins=10, show_woe=False):
    
#     #Empty Dataframe
#     newdata,woedata = pd.DataFrame(), pd.DataFrame()
    
#     #Extract Column Names
#     cols = data.columns
    
#     #Run WOE and IV on all the independent variables
#     for ivars in cols[~cols.isin([target])]:
#         if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
#             binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
#             d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
#         else:
#             d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
#         d0 = d0.astype({"x": str})
#         d = d0.groupby("x", as_index=False, dropna=False).agg({"y": ["count", "sum"]})
#         d.columns = ['Cutoff', 'N', 'Events']
#         d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
#         d['Non-Events'] = d['N'] - d['Events']
#         d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
#         d['WoE'] = np.log(d['% of Non-Events']/d['% of Events'])
#         d['IV'] = d['WoE'] * (d['% of Non-Events']-d['% of Events'])
#         d.insert(loc=0, column='Variable', value=ivars)
#         print("Information value of " + ivars + " is " + str(round(d['IV'].sum(),6)))
#         temp =pd.DataFrame({"Variable" : [ivars], "IV" : [d['IV'].sum()]}, columns = ["Variable", "IV"])
#         newdata=pd.concat([newdata,temp], axis=0)
#         woedata=pd.concat([woedata,d], axis=0)

#         #Show WOE Table
#         if show_woe == True:
#             print(d)
#     return newdata, woedata
# iv, woe = iv_woe(data = mydata, target = 'admit', bins=10, show_woe = True)
# iv, woe