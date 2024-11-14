import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_regression_metrics(y_true, y_pred):
    """Calculate multiple regression evaluation metrics."""
    
    # Calculate RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate R2 (R-squared)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate RPIQ (Relative Prediction Interval Quality)
    # y_std = np.std(y_true)
    # rpiq = 1 - (rmse / y_std)
    
    # Calculate MAE (Mean Absolute Error)
    mae = np.mean(np.abs(y_true - y_pred))
    

    def rpiq_metric(y_real, y_pred):
     # Calculate quartiles Q1 and Q3
     q1 = np.percentile(y_real, 25)
     q3 = np.percentile(y_pred, 75)

     # Calculate RMSE
     rmse = np.sqrt(mean_squared_error(y_real, y_pred))

     # Calculate the ratio of the difference between Q3 and Q1 to RMSE
     ratio = (q3 - q1) / rmse

     return ratio
    
    rpiq = rpiq_metric(y_true, y_pred)
    
    # Calculate CCC (Concordance Correlation Coefficient)
    def concordance_correlation_coefficient(y_real, y_pred):
        # Raw data
        dct = {
            'y_real': y_real,
            'y_pred': y_pred
        }
        df = pd.DataFrame(dct)
        # Remove NaNs
        df = df.dropna()
        # Pearson product-moment correlation coefficients
        y_real = df['y_real']
        y_pred = df['y_pred']
        cor = np.corrcoef(y_real, y_pred)[0][1]
        # Means
        mean_real = np.mean(y_real)
        mean_pred = np.mean(y_pred)
        # Population variances
        var_real = np.var(y_real)
        var_pred = np.var(y_pred)
        # Population standard deviations
        sd_real = np.std(y_real)
        sd_pred = np.std(y_pred)
        # Calculate CCC
        numerator = 2 * cor * sd_real * sd_pred
        denominator = var_real + var_pred + (mean_real - mean_pred)**2

        return numerator / denominator
    
    ccc = concordance_correlation_coefficient(y_true, y_pred)
    
    return rmse, r2, rpiq, mae, ccc



def normalize_dataframe(df):
    # Copy the original DataFrame to avoid modifying the original data
    normalized_df = df.copy()
    
    # Calculate the minimum and maximum values for each column
    min_values = df.min()
    max_values = df.max()
    
    # Normalize the data of each column
    for column in df.columns:
        # Calculate the range for normalization
        data_range = max_values[column] - min_values[column]
        
        # Normalize the data in the column
        normalized_df[column] = (df[column] - min_values[column]) / data_range
    
    return normalized_df

from sklearn.model_selection import train_test_split

def create_stratified_splits(X, y, test_size=0.2, n_grp=5, random_state=None):
    # Binning the target variable
    grp = pd.qcut(y, n_grp, labels=False)
    
    # Perform stratified random sampling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        stratify=grp, random_state=random_state)
    
    return X_train, X_test, y_train, y_test


def denormalize_dataframe(normalized_df, original_df):
    # Copy the normalized DataFrame to avoid modifying it
    denormalized_df = normalized_df.copy()
    
    # Calculate the minimum and maximum values for each column in the original DataFrame
    min_values = original_df.min()
    max_values = original_df.max()
    
    # Denormalize the data of each column
    for column in original_df.columns:
        # Calculate the range for denormalization
        data_range = max_values[column] - min_values[column]
        
        # Denormalize the data in the column
        denormalized_df[column] = normalized_df[column] * data_range + min_values[column]
    
    return denormalized_df
