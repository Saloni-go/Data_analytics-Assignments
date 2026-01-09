import pandas as pd
import numpy as np
from scipy import stats
from typing import Union, List, Optional 

_original_df=None 
_working_df=None
MISSING_VALUES = ["", "?", "na", "n/a", "none", "null", "nan", "missing", "-", "undefined"]

# RESET DATASET
def reset_dataset():
    global _original_df, _working_df
    if _original_df is None:
        print("No original dataset available.")
        return None
        
    _working_df = _original_df.copy(deep=True)
    print("Dataset reset to original.")
    print(f"Dataset shape: {_working_df.shape}")
    return _working_df

    
# LOAD DATASET
def load_dataset(path):
    global _original_df,_working_df
    try:
        _original_df = pd.read_csv(path)
        _working_df = _original_df.copy(deep=True)
        print(f"Dataset loaded successfully with shape: {_original_df.shape}")
        return _original_df, _working_df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None


# FILL SELECTED ROWS WITH MEAN
def fill_missing_with_mean(df, col_name, rows=None):
    if col_name not in df.columns:
        print(f"Column '{col_name}' not found.")
        return df
    
    if not pd.api.types.is_numeric_dtype(df[col_name]):
        print(f"Column '{col_name}' is not numeric. Cannot fill with mean.")
        return df

     # If no specific rows provided, fill all missing values
    if rows is None:
        missing_mask = df[col_name].isna()
        rows = df[missing_mask]
    if rows.empty:
        print(f"No missing values found in '{col_name}'.")
        return df    
    mean_val = df[col_name].mean()

    # Show preview of what will be changed
    print(f"\nProposed action: Fill {len(rows)} rows in '{col_name}' with mean: {mean_val:.4f}")
    print("Sample of rows that will be affected:")
    print(rows.head())  # Show first few rows
    
    # Ask for confirmation
    confirmation = input("\nDo you want to implement this change? (yes/no): ").strip().lower()
    
    if confirmation in ['yes', 'y']:
        # Fill only the detected rows
        df.loc[rows.index, col_name] = mean_val
        print(f"✓ Filled {len(rows)} rows in '{col_name}' with mean: {mean_val:.4f}")
        
        global _working_df
        _working_df = df
    else:
        print("✗ Change cancelled. No rows were modified.")

    return df
    


# FILL SELECTED ROWS WITH MEDIAN
def fill_missing_with_median(df, col_name, rows=None):
    if col_name not in df.columns:
        print(f"Column '{col_name}' not found.")
        return df
    
    if not pd.api.types.is_numeric_dtype(df[col_name]):
        print(f"Column '{col_name}' is not numeric. Cannot fill with median.")
        return df
    
    # If no specific rows provided, fill all missing values
    if rows is None:
        missing_mask = df[col_name].isna()
        rows = df[missing_mask]
    if rows.empty:
        print(f"No missing values found in '{col_name}'.")
        return df    

    median_val = df[col_name].median()

     # Show preview of what will be changed
    print(f"\nProposed action: Fill {len(rows)} rows in '{col_name}' with mean: {median_val:.4f}")
    print("Sample of rows that will be affected:")
    print(rows.head())  # Show first few rows
    
    # Ask for confirmation
    confirmation = input("\nDo you want to implement this change? (yes/no): ").strip().lower()
    
    if confirmation in ['yes', 'y']:
        # Fill only the detected rows
        df.loc[rows.index, col_name] = median_val
        print(f"✓ Filled {len(rows)} rows in '{col_name}' with mean: {median_val:.4f}")
        
        global _working_df
        _working_df = df
    else:
        print("✗ Change cancelled. No rows were modified.")
    

    return df

# REMOVE MISSING VALUES BY DELETING THOSE ROWS
def remove_missing_values(df, rows, col_name):
    if rows.empty:
        print("No rows to delete.")
        return df
     
    choice = input(f"Do you want to delete these rows? (yes/no/range/indices): ").strip().lower()

    if choice == "yes":
        df = df.drop(rows.index).reset_index(drop=True)
        print(f"Deleted all rows detected. New shape: {df.shape}")
        print("\n")
    elif choice == "range":
        try:
            start = int(input("Enter start index (inclusive): "))
            end = int(input("Enter end index (inclusive): "))
            indices_to_delete = rows.index[(rows.index >= start) & (rows.index <= end)]
            df = df.drop(indices_to_delete).reset_index(drop=True)
            print(f"Deleted rows from {start} to {end}. New shape: {df.shape}")
            print("\n")
        except ValueError:
            print("Invalid input. Please enter valid integers.")
    elif choice == "indices":
        try:
            idx_list = input("Enter indices to delete (comma separated): ")
            indices = [int(x.strip()) for x in idx_list.split(",")]
            df = df.drop(indices).reset_index(drop=True)
            print(f"Deleted rows {indices}. New shape: {df.shape}")
            print("\n")
        except ValueError:
            print("Invalid input. Please enter valid integers separated by commas.")
    else:
        print("No rows deleted.")

    global _working_df
    _working_df = df
    return df

    

# INTERACTIVE NEXT STEP
def interactive_next_step(df, rows, col_name, message=""):
    if rows.empty:
        print(f"No rows found for {message}.")
        return df
    
    print(f"Rows detected for {message}:")
    print(rows)

    # Ask user for action
    choice = input(
        f"\nChoose action for column '{col_name}' ({message}):\n"
        " - 'mean'   → replace with column mean\n"
        " - 'median' → replace with column median\n"
        " - 'delete' → remove these rows\n"
        "Enter choice: "
    ).strip().lower()

    if choice == "mean":
        df = fill_missing_with_mean(df, col_name, rows)
    
    elif choice == "median":
        df = fill_missing_with_median(df, col_name, rows)

    else:
        df = remove_missing_values(df, rows, col_name)

    global _working_df
    _working_df = df
    return df


# REMOVING MISSING VALUES
def check_missing_values(df, col_name):
    if col_name not in df.columns:
        print(f"Column '{col_name}' not found.")
        return df 

    print(f"Checking column: {col_name}")
    col = df[col_name]

    # traverse row by row
    invalid_indices = []
    for i, val in enumerate(col):
        if pd.isna(val):
            invalid_indices.append(i)
        elif isinstance(val, str) and val.strip() == "": 
            invalid_indices.append(i) 
        elif isinstance(val, str) and val.lower() in MISSING_VALUES:
            invalid_indices.append(i)

    if invalid_indices:
        rows = df.loc[invalid_indices]
        df = interactive_next_step(df, rows, col_name, message=f"missing/invalid values in '{col_name}'")
    else:
        print(f"No missing or invalid values found in column '{col_name}'.")

    print("Check complete.")
    global _working_df
    _working_df = df
    return df


# CHECK FOR OUTLIERS - IQR
def detect_outliers_iqr(df, column):
    if column not in df.columns:
        print(f"Column '{column}' not found.")
        return df

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
   
    lower_bound = Q1 - 5 * IQR
    upper_bound = Q3 + 5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    if outliers.empty:
        print(f"No outliers found in column '{column}'.")
    else:
        print(f"Outliers found in column '{column}':")
        print(outliers)
        df = interactive_next_step(df, outliers, column, message=f"outliers in '{column}'")

    global _working_df
    _working_df = df
    return df


# Z-SCORE METHOD
def detect_outliers_zscore(df, column, threshold=3.5):
    if column not in df.columns:
        print(f"Column '{column}' not found.")
        return df
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        print(f"Column '{column}' is not numeric. Z-score can only be applied to numeric columns.")
        return df
    
    z_scores = np.abs(stats.zscore(df[column], nan_policy='omit'))

    # Create a mask for outliers, handling NaN values
    outlier_mask = pd.Series([False] * len(df))
    valid_indices = df[column].dropna().index
    outlier_mask.loc[valid_indices] = z_scores > threshold
    
    outliers = df[outlier_mask]

    if not outliers.empty:
        print(f"Outliers found in column '{column}':")
        print(outliers)
        df = interactive_next_step(df, outliers, column, message=f"Z-score outliers in '{column}'")
    else:
        print(f"No outliers found in column '{column}'.")

    global _working_df
    _working_df = df
    return df


# ENSURING CORRECT DATA TYPES
def ensure_data_type(df, col_name, target_dtype=None):
    if col_name not in df.columns:
        print(f"Column '{col_name}' not found.")
        return df

    current_dtype = df[col_name].dtype
    print(f"Column '{col_name}' current dtype: {current_dtype}")

      # If no target dtype specified, try to infer
    if target_dtype is None:
        if pd.api.types.is_numeric_dtype(df[col_name]):
            print(f"Column '{col_name}' is already numeric.")
            return df
        else:
            target_dtype = 'numeric'
    
    if target_dtype == 'numeric':
        converted = pd.to_numeric(df[col_name], errors='coerce')
        if converted.isna().any():
            print(f"Warning: Some values in '{col_name}' could not be converted to numeric.")
        df[col_name] = converted
        print(f"Column '{col_name}' converted to numeric.")

    elif target_dtype == 'datetime':
        converted = pd.to_datetime(df[col_name], errors='coerce')
        if converted.isna().any():
            print(f"Warning: Some values in '{col_name}' could not be converted to datetime.")
        df[col_name] = converted
        print(f"Column '{col_name}' converted to datetime.")

    else:
        try:
            df[col_name] = df[col_name].astype(target_dtype)
            print(f"Column '{col_name}' converted to {target_dtype}.")
        except Exception as e:
            print(f"Error converting column '{col_name}' to {target_dtype}: {e}")
    
    print(f"Final dtype of column '{col_name}': {df[col_name].dtype}")
    global _working_df
    _working_df = df
    return df

# REMOVE DUPLICATES
# REMOVE DUPLICATES
def remove_duplicates(df, col_name=None):
    if col_name is not None:
        if isinstance(col_name, str):
            if col_name not in df.columns:
                print(f"Column '{col_name}' not found.")
                return df
            col_name_list = [col_name]  # Store as list for subset parameter
        else:  # col_name is a list
            for col in col_name:
                if col not in df.columns:
                    print(f"Column '{col}' not found.")
                    return df
            col_name_list = col_name  # Already a list
    else:
        col_name_list = None  # Check all columns for duplicates
        
    # Handle the subset parameter correctly
    duplicates = df[df.duplicated(subset=col_name_list, keep=False)]
    
    if not duplicates.empty:
        if col_name_list:  # If checking specific columns
            duplicates = duplicates.sort_values(by=col_name_list)
        print(f"\n{len(duplicates)} duplicate rows found:")
        print(duplicates)
        
        # Create a display name for the message
        if col_name is None:
            display_name = "all columns"
        elif isinstance(col_name, str):
            display_name = col_name
        else:
            display_name = ", ".join(col_name)
            
        df = interactive_next_step(df, duplicates, display_name, message=f"duplicates in '{display_name}'")
    else:
        if col_name is None:
            print("No duplicates found in the dataset.")
        elif isinstance(col_name, str):
            print(f"No duplicates found in column '{col_name}'.")
        else:
            print(f"No duplicates found in columns {col_name}.")
            
    global _working_df
    _working_df = df
    return df

if __name__ == "__main__":
    print("This is the dataCleaning module. Import it to use its functions.")
