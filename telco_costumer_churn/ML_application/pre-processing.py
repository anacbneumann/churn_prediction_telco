import pandas as pd
import os

# Declare file_path and output_dir
file_path = r'file_path/WA_Fn-UseC_-Telco-Customer-Churn.csv'
output_dir = r'output_path/ML_application'

def preprocess_telco_data(file_path, output_dir):
    """
    Preprocesses the Telco Customer Churn dataset.

    As the dataset was already well-prepared to receive the 
    algorithm and did not require many preprocessing steps, 
    this function only removes the `TotalCharges` column, which 
    has high collinearity. It also verifies the upload and output 
    directories.

    Parameters:
        file_path (str): Path to the input CSV file.
        output_dir (str): Path to the directory to save the preprocessed file.

    Returns:
        None
    """    
    # 1: Load the CSV file
    try:
        df = pd.read_csv(file_path)
        print(f'Successfully loaded: {file_path}')
    except FileNotFoundError:
        print(f'Error: File not found: {file_path}')
        return
    except Exception as e:
        print(f'Error while loading: {e}')
        return

    # 2: Drop unnecessary columns
    df = df.drop(columns=['TotalCharges'], errors='ignore')
    print('Dropped column "TotalCharges" (if existed).')

    # 3: Validate DataFrame
    if df.empty:
        print('DataFrame is empty. Nothing to process.')
        return

    # 4: Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'Created output directory: {output_dir}')

    # 5: Save the processed DataFrame to Excel
    try:
        output_file = os.path.join(output_dir, 'pre_processed.xlsx')
        df.to_excel(output_file, index=False)
        print(f'File saved successfully at: {output_file}')
    except Exception as e:
        print(f'Error while saving the file: {e}')

# Applying the function:
preprocess_telco_data(file_path, output_dir)
