"""
Author: Livin Albert
Purpose: Define the static file names
Action: Update the file name based on requirement
    utils contains the function which
    are required for feature curation step
    and data insertation
"""

from pathlib import Path
from typing import List

import os
import pandas as pd
import numpy as np
import glob

from sklearn.feature_selection import f_classif
from scipy.stats import chi2_contingency, pearsonr, pointbiserialr
from typing import List
import re

def combine_queries(
    query_path: Path, query_list: list, customer_list_table: str, days: str, time_window: str
) -> str:
    """
    Combine SQL files from SQL directory into single SQL statement

        Parameters:
            query_path (Path): _description_
            query_list (List[str]): _description_

        Returns:
            _type_: _description_
    """

    combined_ctes = f"""WITH customers AS (
      SELECT *
      FROM `{customer_list_table}`
    ) \n\n"""

    final_query_select = """
    SELECT
    customers.*, \n """

    final_query_from = """
    FROM customers """

    cte_names = []

    for cte in query_list:
        with open(f"{query_path}/{cte}", "r") as f:
            query = f.read()
            query = query.replace("{{table_name}}", str(customer_list_table))
        
        cte= re.sub(r'^\d+_','',cte)
        cte = cte.replace(".sql", "")
        cte_names.append(cte)
        
        if (cte!='combination_features'):
            cte_query = f""" ----------------------#{cte}--------------------------------------------------,\n,{cte} AS (
            {query}) \n\n"""
            
            combined_ctes += cte_query

            final_query_select += f"{cte}.* EXCEPT(ID_COL, DATE_COL ), \n" # REPLACE THE ID AND DATE WITH NON-FEATURE COLUMN
            final_query_from += f"""\nJOIN {cte} ON customers.ID_COL={cte}.ID_COL
                                        AND customers.DATE_COL={cte}.DATE_COL """
        else:
            final_query_select += f"{query} \n"
     

    combined_query = f"""
    {combined_ctes} 
    {final_query_select}
    {final_query_from}
    """
    return combined_query

def create_table(query: str, table_name: str) -> None:
    """
    Create a big query table from a query

        Parameters:
            query : A sql query to create a table from
            table_name : name of the table to be created
            project : bigquery project ID
            credentials : Google credentials

        Returns:
            Creates a dataset in BQ
    """
    bqclient = bigquery.Client()

    job_config = bigquery.QueryJobConfig(destination=f"{table_name}")
    job_config.write_disposition = "WRITE_TRUNCATE"  # Overwrite table if it exists

    job = bqclient.query(query, job_config=job_config)
    job.result()

    print(table_name + " -complete!")
    
def left_join_dataframes(dataframes, join_column):
    """
    Performs a left join on a list of DataFrames based on the specified column.
    - dataframes (list): A list of DataFrames to join.
    - join_column (str): The column name to use for joining.
    """
    result = dataframes[0].copy()
    for df in dataframes[1:]:
        try:
            if join_column not in df.columns:
                print(f"Warning: Column '{join_column}' not found in DataFrame from {df.name}")
                continue  
            result = pd.merge(result, df, how='left', on=join_column)
        except Exception as e: 
            print(f"Error joining DataFrame: {e}")
    return result

def read_join_csvs(directory_path, join_column_name):
    """
    Reads CSV files one-by-one from a directory, performs left join based on a column.
    - directory_path (str): Path to the directory containing CSV files.
    - join_column_name (str): The column name to use for joining.
    """
    dataframes = []
    for filename in glob.glob(str(directory_path) + "/*.csv"):
        try:
            df = pd.read_csv(filename)
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading file '{filename}': {e}")
    return left_join_dataframes(dataframes, join_column_name)


def df_null_values(df_raw, statistics_dict):
    """
    Extract statistical measures for null value contains column
    - df_raw (dataframe): Actual dataframe 
    - statistics_dict : Appropriate statistical measures for each column
    - df_out: Datafrmae contains statistical measures for each null value contians column
    """
    # initializing the dataframe for null columns to add the statistical measures
    df_out = pd.DataFrame({'Features': []
                     ,'Categories': []
                     ,'Values': []})
    cols_set = set(df_out['Features'])
    
    # Find the statistical measures for all null features
    for k, v in statistics_dict.items():
        if k == 'Mean':
            fill_value = list(df_raw[v].mean().round(2))
        elif k == 'Median':
            fill_value = list(df_raw[v].median().round(2))
        elif k == 'Mode':
            fill_value = []
            for i in range(len(v)):
                fill_value.append(df_raw[v[i]].mode()[0])
        elif k == 'Zero':
            fill_value = [0 for i in range(len(v))]
                    
        for i in range(len(v)):
            if i in cols_set:
                continue
            df_out = df_out.append(pd.Series((v[i], k, fill_value[i]), index=df.columns), ignore_index=True)
            
    return df_out

def fill_null_values(df, df_null_features):
    """
    Fill nullvalue with appropriate values
    - df : Table need to replace with statisticl measure value where null value present
    - df_null_features
    """
    # Dictionary of null features with values    
    null_features_value_dict = dict(zip(df_null_features['Features'], df_null_features['Values']))

    # Null values columns found in table
    null_cols = [col for col in df.columns if df[col].isnull().any()]
      
    # iterating through all the null columns and filling the null values
    for feature in null_cols:

    # null values columns filling   
        if feature in null_features_value_dict: 
            try:
                df[feature].fillna(null_features_value_dict[feature], inplace=True) 
            except:
                df[feature].fillna(int(null_features_value_dict[feature]), inplace=True)  
        else:
            continue
    return df



def feature_types(df: pd.DataFrame):
        """
        input: Consolidated dataframe
        Returns: Different types of feature list
        """

        list_categorical_features = []
        list_string_features = []
        list_numerical_features = []
        list_uncategorised_features = []

        if df is not None:
            for column in list(df.columns):
                # Not a target variable of intentionally excluded
                if column not in list_churn_flags:
                    datatype = df[column].dtype
                    count_distinct = len(df[column].unique())

                    if datatype in [str, object]:
                        list_string_features.append(column)
                    elif count_distinct == 2:
                        list_categorical_features.append(column)
                    elif datatype in [float, int, "Int64"]:
                        list_numerical_features.append(column)
                    else:
                        list_uncategorised_features.append(column)

        return (
            list_categorical_features,
            list_string_features,
            list_numerical_features,
            list_uncategorised_features,
        )

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop empty rows

        Parameters:
            df (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: Cleaned dataset


    """

    assert isinstance(df, pd.DataFrame)

    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)

    return df[indices_to_keep].astype(np.float64)


def calculate_null_percentage(df: pd.DataFrame, columns: List):
    """
    Calculate the NULL percentage

        Parameters:
            df (pd.DataFrame): _description_
            columns (List[str]): _description_

        Returns:
            _type_: _description_

        Todo:
            Update the return type
            Update the description for method and return type
            Update the description for parameters
    """

    total_rows = len(df)
    null_percentages = df[columns].isnull().sum() / total_rows * 100

    return null_percentages


def transform_string_columns(df: pd.DataFrame, string_columns: list) -> pd.DataFrame:
    """
    One-hot encoding

        Parameters:
            df (pd.DataFrame): Dataframe to apply One-hot encoding
            string_columns (List[str]): Columns to apply One-hot encoding

        Returns:
            df_transformed (pd.DataFrame): One-hot encoded dataframe
    """

    df_transformed = pd.get_dummies(df, columns=string_columns)

    return df_transformed

def read_data(
    table_name: str, bqclient: any, environment: str = "prod"
) -> pd.DataFrame:
    """
    Read the data from file system / BiqQuery based on environment

        Parameters:
            table_name (str): _description_
            google_credentials (any): _description_
            environment (str, optional): _description_. Defaults to 'prod'.

        Raises:
            ValueError: _description_

        Returns:
            pd.DataFrame: _description_
    """
    # Check the envrionment is accepted value
    if environment not in ENVIRONMENTS:
        raise ValueError(f"results: status must be one of {ENVIRONMENTS}")

    # Read the file based on the environment
    if environment == "dev":
        df = pd.read_csv(DATA / f"{table_name}.csv")
    else:
        df = bqclient.query(f"SELECT * FROM {table_name}").to_dataframe()

    return df


def read_query(bqclient: any, query: str) -> pd.DataFrame:
    """
    Read data from BigQuery using SQL query

        Parameters:
            google_credentials (any): _description_
            query (str): _description_

        Returns:
            pd.DataFrame: _description_
    """
    df = bqclient.query(query).to_dataframe()

    return df


def write_data(table_name: str, df: pd.DataFrame, client: any, environment: str = "prod") -> None:
    """
    Write data to file system / BigQuery based on environment

        Parameters:
            table_name (str): _description_
            df (pd.DataFrame): _description_
            client (any): _description_
            environment (str, optional): _description_. Defaults to 'prod'.

        Returns:
            _type_: _description_
    """
    # Check the envrionment is accepted value
    if environment not in ENVIRONMENTS:
        raise ValueError(f"results: status must be one of {ENVIRONMENTS}")

    if environment == 'prod':
        df = df.convert_dtypes()

        # Define the config for the write table
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE",
        )

        job = client.load_table_from_dataframe(
            df, table_name, job_config=job_config
        )  # Make an API request.
        job.result()  # Wait for the job to complete.

        table = client.get_table(table_name)  # Make an API request.
        print(
            f"Loaded {table.num_rows} rows and {len(table.schema)} columns to {table_name}."
        )
    else:
        # Save the data to local directory
        df.to_csv(DATA / f"{table_name}.csv", index=False)
        print(f"File Saved at {DATA}")
