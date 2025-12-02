import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


def assign_era(year):
    if year < 1900:
        return 'Pre-1900'
    elif 1900 <= year < 1950:
        return '1900-1949'
    elif 1950 <= year < 2000:
        return '1950-1999'
    elif 2000 <= year < 2020:
        return '2000-2019'
    else:
        return '2020+'

def extract_quadrant(address):
    if pd.isna(address):  
        return 'Other'

    suffix = address.strip()[-2:].upper()


    if suffix in ['NW', 'NE', 'SW', 'SE']:
        return suffix
    else:
        return 'Other'

def replace_land_size(df):
    low_rise_apartment_condo_median = df[
        (df['SUB_PROPERTY_USE'] == 'RE0201') &
        (df['ERA'] == '2020+') &
        (df['COMM_CODE'] == 'SPH') &
        (df['QUADRANT'] == 'SW') &
        (df['LAND_SIZE_SM'] != 0)
    ]['LAND_SIZE_SM'].median()

    # Calculate the median for duplexes (RE0120)
    duplex_median = df[
        (df['SUB_PROPERTY_USE'] == 'RE0120') &
        (df['ERA'] == '2020+') &
        (df['COMM_CODE'] == 'HPK') &
        (df['QUADRANT'] == 'NW') &
        (df['LAND_SIZE_SM'] != 0)
    ]['LAND_SIZE_SM'].median()

    # Replace infinities and NaNs with the respective medians
    df['LAND_SIZE_SM'] = np.where(
        ((df['LAND_SIZE_SM'] == 0) | pd.isna(df['LAND_SIZE_SM'])) &
        (df['SUB_PROPERTY_USE'] == 'RE0201') &
        (df['COMM_CODE'] == 'SPH') &
        (df['QUADRANT'] == 'SW') &
        (df['LAND_USE_DESIGNATION'] == 'MU-1'),
        low_rise_apartment_condo_median,
        df['LAND_SIZE_SM']
    )

    df['LAND_SIZE_SM'] = np.where(
        ((df['LAND_SIZE_SM'] == 0) | pd.isna(df['LAND_SIZE_SM'])) &
        (df['SUB_PROPERTY_USE'] == 'RE0120') &
        (df['COMM_CODE'] == 'HPK') &
        (df['QUADRANT'] == 'NW') &
        (df['LAND_USE_DESIGNATION'] == 'R-C2'),
        duplex_median,
        df['LAND_SIZE_SM']
    )

    return df

def group_land_use(land_use):
    if isinstance(land_use, str):  
        if 'R-' in land_use:
            return 'Residential'
        elif 'M-' in land_use:
            return 'Multi-Residential'
        elif 'C-' in land_use and 'CC-' not in land_use:
            return 'Commercial'
        elif 'I-' in land_use:
            return 'Industrial'
        elif 'S-' in land_use:
            return 'Special Purpose'
        elif 'CC-' in land_use or 'R20' in land_use:
            return 'City Centre Districts'
        elif 'MU-' in land_use:
            return 'Mixed Use'
        elif 'DC' in land_use:
            return 'Direct Control District'
        else:
            return 'Undesignated Road Right-of-Way'
    else:
        return 'Other'  

def fill_land_use_design(df):
    criteria = ['QUADRANT', 'ERA']

    # Group by criteria and calculate the mode of LAND_USE_DESIGNATION
    mode_imputation_values = df.groupby(criteria)['LAND_USE_DESIGNATION'].transform(
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    )

    # Impute missing values using np.where
    df['LAND_USE_DESIGNATION'] = np.where(
        pd.isna(df['LAND_USE_DESIGNATION']),  # Check if LAND_USE_DESIGNATION is NaN
        mode_imputation_values,  # Replace NaN with the mode value
        df['LAND_USE_DESIGNATION']  # Keep the original value if not NaN
    )

    return df


def fill_yoc(df):
    criteria = ['LAND_USE_DESIGNATION', 'QUADRANT', 'COMM_CODE']

    year_imputation_values = df.groupby(criteria)['YEAR_OF_CONSTRUCTION'].median()

    df['YEAR_OF_CONSTRUCTION'] = df.apply(
        lambda row: year_imputation_values.loc[tuple(row[criteria])] if pd.isna(row['YEAR_OF_CONSTRUCTION']) else row[
            'YEAR_OF_CONSTRUCTION'],
        axis=1
    )

    return df

def apply_derived_columns(df):
    df["LOG_LAND_SIZE"] = np.log(df["LAND_SIZE_SM"] + 1)
    df['LAND_USE_GROUPED'] = df['LAND_USE_DESIGNATION'].apply(group_land_use)
    df['QUADRANT'] = df['ADDRESS'].apply(extract_quadrant)
    df = fill_yoc(df)
    df['ERA'] = df['YEAR_OF_CONSTRUCTION'].apply(assign_era)
    df = fill_land_use_design(df)
    df = replace_land_size(df)
    df["LOG_LAND_SIZE"] = np.log(df["LAND_SIZE_SM"] + 1)

    # Return the dataframe with all derived columns
    return df

def predict(data: pd.DataFrame) -> np.ndarray:
    """
    Load your pre-trained model, apply your preprocessing pipeline,
    and call the predict method.
    Returns an array of predicted values based on the input data.
    """
    # Load the pre-trained model and preprocessing pipeline
    try:
        pipeline = joblib.load('preprocessing_pipeline_and_model.pkl')  
    except FileNotFoundError:
        raise FileNotFoundError("Pre-trained model and pipeline not found. Please train and save the model first.")

    data = apply_derived_columns(data)

    predictions = pipeline.predict(data)

    return predictions


def plot_predictions(actual, pred):
    plt.scatter(actual, pred, alpha=0.1)
    plt.plot([0, 17500000], [0, 17500000], color="red")
    plt.xlabel("Actual House Value ($)")
    plt.ylabel("Predicted House Value ($)")
    mae = mean_absolute_error(actual, pred)
    plt.text(300000, 10000, f"MAE = ${mae:.2f}")
    plt.show()


if __name__ == "__main__":
    # might be a good idea to test your predict function here

    data = pd.read_csv('data/yyc_housing_2024.csv')
    X = data.drop(columns=['ASSESSED_VALUE'])


    predictions = predict(data)
    y = data['ASSESSED_VALUE']
    plot_predictions(y, predictions)

    pass