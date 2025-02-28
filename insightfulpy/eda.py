#=================================================================
#                     INSIGHTFULPY.0.1.7    
#=================================================================
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import researchpy as rp
from tableone import TableOne
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
from tabulate import tabulate
import numpy as np
from pandas.api.types import is_datetime64_any_dtype
from scipy import stats
from collections import defaultdict, Counter
import textwrap
from scipy.stats import skew, kurtosis, shapiro, kstest



#=================================================================
#                 COMPARE_COLUMNS FUNCTIONS   
#=================================================================

def compare_df_columns(base_df_name, dataframes):
    base_df = dataframes[base_df_name]
    linked_columns = {}

    for col in base_df.columns:
        for df_name, df in dataframes.items():
            if df_name != base_df_name and col in df.columns:
                linked_columns.setdefault(col, []).append(df_name)

    def get_profile(df, columns, df_name):
        df = df[columns]
        profile_data = []
        for col in df.columns:
            total_rows = len(df)
            missing_values = df[col].isnull().sum()
            missing_percent = (missing_values / total_rows) * 100
            negative_values = (df[col] < 0).sum() if pd.api.types.is_numeric_dtype(df[col]) else 'N/A'
            negative_percent = (negative_values / total_rows) * 100 if isinstance(negative_values, int) else 'N/A'
            if pd.api.types.is_numeric_dtype(df[col]):
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                outliers = df[col][(df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))].count()
                outlier_percent = (outliers / total_rows) * 100
            else:
                outliers = 'N/A'
                outlier_percent = 'N/A'
            data_type = df[col].dtype
            profile_data.append({
                "Dataset": df_name,
                "Column": col,
                "Rows": total_rows,
                "Columns": 1,
                "Missing Values": missing_values,
                "Missing %": f"{missing_percent:.2f}%",
                "Negative Values": negative_values if negative_values != 'N/A' else 'N/A',
                "Negative %": f"{negative_percent:.2f}%" if isinstance(negative_percent, float) else 'N/A',
                "Outliers": outliers if outliers != 'N/A' else 'N/A',
                "Outlier %": f"{outlier_percent:.2f}%" if isinstance(outlier_percent, float) else 'N/A',
                "Data Types": data_type
            })
        return pd.DataFrame(profile_data)

    # Profile of base dataset
    base_profile_df = get_profile(base_df, linked_columns.keys(), base_df_name)

    # Profiles of linked columns in other datasets
    linked_profiles_df = pd.DataFrame()
    for col, dfs in linked_columns.items():
        for df_name in dfs:
            linked_profile_df = get_profile(dataframes[df_name], [col], df_name)
            linked_profiles_df = pd.concat([linked_profiles_df, linked_profile_df], ignore_index=True)

    return base_profile_df, linked_profiles_df

def linked_key(dataframes):
    profile_data = []
    total_rows, total_columns, total_cells = 0, 0, 0
    total_missing, total_negative, total_outliers = 0, 0, 0
    combined_dtypes = {}

    # Loop through each DataFrame for profiling
    for name, df in dataframes.items():
        rows, cols = df.shape
        total_rows += rows
        total_columns += cols
        dataset_cells = df.size
        total_cells += dataset_cells

        # Missing Values
        missing_values = df.isnull().sum().sum()
        total_missing += missing_values
        missing_percentage = (missing_values / dataset_cells) * 100

        # Data Types Count
        dtypes_count = df.dtypes.value_counts().to_dict()
        for dtype, count in dtypes_count.items():
            dtype_str = str(dtype)
            combined_dtypes[dtype_str] = combined_dtypes.get(dtype_str, 0) + count
        dtype_summary = ', '.join([f'{dtype}: {count}' for dtype, count in dtypes_count.items()])

        # Negative Values
        negative_values = df.select_dtypes(include=[np.number]).lt(0).sum().sum()
        total_negative += negative_values
        negative_percentage = (negative_values / dataset_cells) * 100

        # Outlier Detection using IQR Method
        numeric_df = df.select_dtypes(include=[np.number])
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).sum().sum()
        total_outliers += outliers
        outlier_percentage = (outliers / dataset_cells) * 100

        # Append to profile data
        profile_data.append({
            'Dataset': name,
            'Rows': rows,
            'Columns': cols,
            'Missing Values': missing_values,
            'Missing %': round(missing_percentage, 2),
            'Negative Values': negative_values,
            'Negative %': round(negative_percentage, 2),
            'Outliers': outliers,
            'Outlier %': round(outlier_percentage, 2),
            'Data Types': dtype_summary
        })

    # Total summary row
    total_missing_percentage = (total_missing / total_cells) * 100
    total_negative_percentage = (total_negative / total_cells) * 100
    total_outlier_percentage = (total_outliers / total_cells) * 100
    total_dtype_summary = ', '.join([f'{dtype}: {count}' for dtype, count in combined_dtypes.items()])

    profile_data.append({
        'Dataset': 'Total',
        'Rows': total_rows,
        'Columns': total_columns,
        'Missing Values': total_missing,
        'Missing %': round(total_missing_percentage, 2),
        'Negative Values': total_negative,
        'Negative %': round(total_negative_percentage, 2),
        'Outliers': total_outliers,
        'Outlier %': round(total_outlier_percentage, 2),
        'Data Types': total_dtype_summary
    })

    # Display profile
    profile_df = pd.DataFrame(profile_data)
    display(profile_df)  # Use print(profile_df) if not in Jupyter Notebook

    # Identify link columns
    column_map = defaultdict(list)
    for name, df in dataframes.items():
        for column in df.columns:
            column_map[column].append(name)

    link_columns = {col: dfs for col, dfs in column_map.items() if len(dfs) > 1}

    # Display link columns
    if link_columns:
        print("\n### Link Columns (Common Columns Across DataFrames):\n")
        for column, dfs in link_columns.items():
            print(f"- {column}: {', '.join(dfs)}")
    else:
        print("\nNo common link columns found across the DataFrames.")

def display_key_columns(base_df_name, dataframes):
    base_df = dataframes[base_df_name]
    linked_columns = {}

    for col in base_df.columns:
        for df_name, df in dataframes.items():
            if df_name != base_df_name and col in df.columns:
                linked_columns.setdefault(col, []).append(df_name)

    table_data = [(col, ', '.join(dfs)) for col, dfs in linked_columns.items()]
    print(tabulate(table_data, headers=["Column", "Linked DataFrames"], tablefmt="pipe"))


def interconnected_outliers(df, outlier_cols):
    outlier_rows = defaultdict(list)
    for col in outlier_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        for idx in df[(df[col] < lower_bound) | (df[col] > upper_bound)].index:
            outlier_rows[idx].append(col)
    
    # Count occurrences of each set of columns for outlier rows where multiple columns are outliers
    column_set_counter = Counter(tuple(sorted(cols)) for cols in outlier_rows.values() if len(cols) > 1)
    
    print(f"\nTotal Interconnected Outliers: {sum(len(cols) > 1 for cols in outlier_rows.values())}")
    print("Column Set Outlier Frequency:")
    for columns, count in column_set_counter.items():
        print(f"  Columns {', '.join(columns)}: {count} times")
    
    # Filtering and returning rows that are outliers in more than one column
    interconnected_outlier_rows = [idx for idx, cols in outlier_rows.items() if len(cols) > 1]
    
    # Avoid printing interconnected outliers section
    if interconnected_outlier_rows:
        return df.loc[interconnected_outlier_rows]
    else:
        return pd.DataFrame()

def grouped_summary(data, groupby=None):
    # Separate categorical and numerical columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = data.select_dtypes(include=['number']).columns.tolist()

    # Exclude columns with mixed data types
    mixed_cols = [col for col in data.columns if data[col].map(type).nunique() > 1]
    data = data.drop(columns=mixed_cols)

    # Ensure p-value calculation is enabled only when a groupby column is specified
    pval_flag = True if groupby else False

    # Generate TableOne
    table = TableOne(data, categorical=categorical_cols, groupby=groupby, pval=pval_flag, isnull=True)

    # Print summary information
    if groupby:
        print(f"=== Summary Grouped by '{groupby}' ===")
    else:
        print("=== Summary (No Grouping) ===")

    return table


#=================================================================
#                  BASIC CHECK-UP AND STATS FUNCTIONS
#=================================================================

def calc_stats(data):
    return {
        'Count': data.count(),
        'Mean': data.mean(),
        'Trimmed Mean': iqr_trimmed_mean(data),
        'MAD': mad(data),
        'Std': data.std(),
        'Min': data.min(),
        '25%': data.quantile(0.25),
        '50%': data.median(),
        '75%': data.quantile(0.75),
        'Max': data.max(),
        'Mode': data.mode()[0] if not data.mode().empty else 'N/A',
        'Range': data.max() - data.min(),
        'IQR': data.quantile(0.75) - data.quantile(0.25),
        'Variance': data.var(),
        'Skewness': data.skew(),
        'Kurtosis': data.kurt()
    }


def iqr_trimmed_mean(data):
    q1, q3 = np.percentile(data.dropna(), [25, 75])
    iqr = q3 - q1
    return data[(data >= q1 - 1.5 * iqr) & (data <= q3 + 1.5 * iqr)].mean()


def mad(data):
    return np.mean(np.abs(data - data.mean()))



def comp_cat_analysis(df, missing_df=False):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    result = []

    for col in cat_cols:
        data = df[col].astype(str).dropna()  # Convert categorical data to string
        count = data.count()
        unique_count = data.nunique()
        mode = data.mode().iloc[0] if not data.mode().empty else np.nan
        mode_freq = data.value_counts().iloc[0] if not data.value_counts().empty else np.nan
        mode_percent = (mode_freq / count) * 100 if count > 0 else 0

        missing_percentage = 100 * (df[col].isnull().sum() / len(df))
        data_type = df[col].dtype.name  # Keep data type as string

        result.append({
            'Index': df.columns.get_loc(col),
            'Column': col,
            'DataType': data_type,
            'Count': count,
            'Missing_Percentage': missing_percentage,
            'Unique_Count': unique_count,
            'Mode': mode,
            'Mode Frequency': mode_freq,
            'Mode %': mode_percent
        })

    summary_df = pd.DataFrame(result)

    if missing_df:
        # Split into missing and non-missing DataFrames
        missing_df_part = summary_df[summary_df['Missing_Percentage'] > 0]
        non_missing_df_part = summary_df[summary_df['Missing_Percentage'] == 0]

        # Simplify sorting by avoiding the DataType column if it causes issues
        missing_df_part = missing_df_part.sort_values(by=['Missing_Percentage'])
        non_missing_df_part = non_missing_df_part.sort_values(by=['Column'])

        return missing_df_part, non_missing_df_part

    return summary_df

def comp_num_analysis(df, missing_df=False, outlier_df=False, outlier_df_values=False):
    num_cols = df.select_dtypes(include=[np.number]).columns
    result = []

    for col in num_cols:
        data = df[col].dropna()
        count = data.count()
        unique_count = data.nunique()
        mean = data.mean()
        std = data.std()
        min_val = data.min()
        q1 = data.quantile(0.25)
        median = data.median()
        q3 = data.quantile(0.75)
        max_val = data.max()
        mode = data.mode().iloc[0] if not data.mode().empty else np.nan
        value_range = max_val - min_val
        iqr = q3 - q1
        variance = data.var()
        skewness = skew(data)
        kurt = kurtosis(data)

        # Normality Test Selection
        if count >= 3:
            if count <= 5000:
                stat, p_value = shapiro(data)
                test_used = 'Shapiro-Wilk'
            else:
                stat, p_value = kstest(data, 'norm', args=(mean, std))
                test_used = 'Kolmogorov-Smirnov'
        else:
            stat, p_value, test_used = np.nan, np.nan, 'Not enough data'

        # Outlier detection using IQR
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outliers_count = outliers.count()
        outliers_distinct = outliers.nunique()
        outliers_percent = (outliers_count / count) * 100 if count > 0 else 0

        # Outlier Values only included when outlier_df is False and outlier_df_values is True
        if outlier_df is False and outlier_df_values is True:
            outlier_values = outliers.tolist()
        else:
            outlier_values = np.nan  # Hide Outlier Values if outlier_df=True

        # Negative values statistics
        negative_values = data[data < 0]
        negative_count = negative_values.count()
        negative_distinct = negative_values.nunique()
        negative_percent = (negative_count / count) * 100 if count > 0 else 0

        missing_percentage = 100 * (df[col].isnull().sum() / len(df))
        data_type = data.dtype

        result_entry = {
            'Index': df.columns.get_loc(col),
            'Column': col,
            'DataType': data_type,
            'Count': count,
            'Missing_Percentage': missing_percentage,
            'Unique_Count': unique_count,
            'Min': min_val,
            'Q1': q1,
            '50% (Median)': median,
            'Q3': q3,
            'Max': max_val,
            'Mode': mode,
            'Range': value_range,
            'IQR': iqr,
            'Lower Bound': lower_bound,
            'Upper Bound': upper_bound,
            'Total Distinct': unique_count,
            'Outliers Distinct': outliers_distinct,
            'Outliers Count': outliers_count,
            'Outliers %': outliers_percent,
            'Negative Count': negative_count,
            'Negative Distinct': negative_distinct,
            'Negative %': negative_percent,
            'Mean': mean,
            'Variance': variance,
            'Std': std,
            'Skewness': skewness,
            'Kurtosis': kurt,
            'Normality Test': test_used,
            'Normality Statistic': stat,
            'Normality p-value': p_value
        }

        # Include Outlier Values only when outlier_df is False and outlier_df_values is True
        if outlier_df is False and outlier_df_values is True:
            result_entry['Outlier Values'] = outlier_values

        result.append(result_entry)

    summary_df = pd.DataFrame(result)

    # Handle missing_df logic
    if missing_df:
        missing_df_part = summary_df[summary_df['Missing_Percentage'] > 0]
        non_missing_df_part = summary_df[summary_df['Missing_Percentage'] == 0]

        missing_df_part = missing_df_part.sort_values(by=['DataType', 'Missing_Percentage'])
        non_missing_df_part = non_missing_df_part.sort_values(by=['DataType'])

        return missing_df_part, non_missing_df_part

    # Handle outlier_df logic
    if outlier_df:
        outlier_df_part = summary_df[summary_df['Outliers Count'] > 0]
        non_outlier_df_part = summary_df[summary_df['Outliers Count'] == 0]

        outlier_df_part = outlier_df_part.sort_values(by=['DataType', 'Outliers %'])
        non_outlier_df_part = non_outlier_df_part.sort_values(by=['DataType'])

        return outlier_df_part, non_outlier_df_part

    return summary_df


def detect_mixed_data_types(data):
    mixed_columns = []
    for col in data.columns:
        # Extract unique data types while ignoring NaNs (read-only operation)
        unique_types = {type(val).__name__ for val in data[col].dropna().values}
        # If more than one type is found, it's mixed
        if len(unique_types) > 1:
            mixed_columns.append([col, ", ".join(sorted(unique_types))])
    # If no mixed data types found
    if not mixed_columns:
        return "No mixed data types detected!"
    # Format output as a table using tabulate
    table = tabulate(mixed_columns, headers=["Column Name", "Detected Data Types"], tablefmt="pipe")
    return print(table)
                
def missing_inf_values(df, missing=False, inf=False, df_table=False):
    total_entries = df.shape[0]
    
    if not missing and not inf:
        missing = inf = True
    
    results = []

    if missing:
        missing_summary = pd.DataFrame({
            'Data Type': df.dtypes,
            'Missing Count': df.isna().sum(),
            'Missing Percentage': (df.isna().sum() / total_entries) * 100
        }).sort_values(by='Missing Percentage', ascending=False)
        missing_summary = missing_summary[missing_summary['Missing Count'] > 0]

        if df_table:
            results.append(missing_summary)
        else:
            print("Missing Values Summary:")
            print(missing_summary if not missing_summary.empty else "No missing values found.")

    if inf:
        infinite_summary = pd.DataFrame({
            'Data Type': df.dtypes,
            'Positive Infinite Count': (df == np.inf).sum(),
            'Positive Infinite Percentage': ((df == np.inf).sum() / total_entries) * 100,
            'Negative Infinite Count': (df == -np.inf).sum(),
            'Negative Infinite Percentage': ((df == -np.inf).sum() / total_entries) * 100
        }).sort_values(by='Positive Infinite Percentage', ascending=False)
        infinite_summary = infinite_summary[(infinite_summary['Positive Infinite Count'] > 0) | (infinite_summary['Negative Infinite Count'] > 0)]

        if df_table:
            results.append(infinite_summary)
        else:
            print("\nInfinite Values Summary:")
            print(infinite_summary if not infinite_summary.empty else "No infinite values found.")

    if df_table:
        return pd.concat(results) if results else None  # Return a single DataFrame

def columns_info(title, data):
    print(f"\n======== {title}: ===========\n")
    print(f"{'Index':<5} {'Col Index':<10} {'Attribute':<30} {'Data Type':<15} {'Range':<30} {'Distinct Count'}")
    print(f"{'-'*5} {'-'*10} {'-'*30} {'-'*15} {'-'*30} {'-'*15}")
    # Sort columns by data type
    sorted_cols = sorted(data.columns, key=lambda col: str(data[col].dtype))
    for i, col in enumerate(sorted_cols, 1):
        col_index = data.columns.get_loc(col)  # Get actual column index
        dtype, distinct = str(data[col].dtype), data[col].nunique()
        rng = f"{data[col].min()} - {data[col].max()}" if dtype in ["int64", "float64","int8","int16","int32","float16","float32"] else "N/A"
        print(f"{i:<5} {col_index:<10} {col:<30} {dtype:<15} {rng:<30} {distinct}")

def cat_high_cardinality(data, threshold=20):
    high_cardinality_cols = [col for col in data.select_dtypes(include=['object', 'category'])
                             if data[col].nunique() > threshold]
    print("high_cardinality_columns")
    return high_cardinality_cols        

def analyze_data(data):
    num_res, cat_res = [], []
    num_cols = data.select_dtypes(include=["int64", "float64","int8","int16","int32","float16","float32"]).columns
    cat_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in num_cols:
        stats = rp.summary_cont(data[col].dropna())
        stats["Variable"] = col
        num_res.append(stats)    
    for col in cat_cols:
        stats = rp.summary_cat(data[col])
        stats["Variable"] = col
        cat_res.append(stats)
    if num_res:
        print("=== Numerical Analysis ===")
        print(pd.concat(num_res, ignore_index=True).to_markdown(tablefmt="pipe"))
    if cat_res:
        print("\n=== Categorical Analysis ===")
        print(pd.concat(cat_res, ignore_index=True).to_markdown(tablefmt="pipe"))

def num_summary(data):
    num_cols = data.select_dtypes(include='number').columns
    if not num_cols.any():
        print("No numerical columns found.")
        return pd.DataFrame()
    return pd.DataFrame({
        col: {
            'Count': data[col].count(),
            'Unique': data[col].nunique(),
            'Mean': round(data[col].mean(), 4),
            'Std': round(data[col].std(), 4),
            'Min': round(data[col].min(), 4),
            '25%': round(data[col].quantile(0.25), 4),
            '50%': round(data[col].median(), 4),
            '75%': round(data[col].quantile(0.75), 4),
            'Max': round(data[col].max(), 4),
            'Mode': data[col].mode()[0] if not data[col].mode().empty else 'N/A',
            'Range': round(data[col].max() - data[col].min(), 4),
            'IQR': round(data[col].quantile(0.75) - data[col].quantile(0.25), 4),
            'Variance': round(data[col].var(), 4),
            'Skewness': round(data[col].skew(), 4),
            'Kurtosis': round(data[col].kurt(), 4),
            'Shapiro-Wilk Stat': round(stats.shapiro(data[col])[0], 4),
            'Shapiro-Wilk p-value': round(stats.shapiro(data[col])[1], 4)
        } for col in num_cols
    }).T

def cat_summary(data):
    cat_cols = data.select_dtypes(include=['object', 'category']).columns
    if not cat_cols.any():
        print("No categorical columns found.")
        return pd.DataFrame()
    return pd.DataFrame({
        col: {
            'Count': data[col].count(),
            'Unique': data[col].nunique(),
            'Top': data[col].mode()[0] if not data[col].mode().empty else 'N/A',
            'Freq': data[col].value_counts().iloc[0] if not data[col].value_counts().empty else 'N/A',
            'Top %': f"{(data[col].value_counts().iloc[0] / data[col].count()) * 100:.2f}%"
        } for col in cat_cols
    }).T

def calculate_skewness_kurtosis(data):
    # Select only numerical columns
    numerical_cols = data.select_dtypes(include=['number']).columns
    # Compute skewness and kurtosis
    skewness = data[numerical_cols].skew()
    kurtosis = data[numerical_cols].kurt()
    # Create a summary DataFrame
    summary = pd.DataFrame({'Skewness': skewness, 'Kurtosis': kurtosis})
    return summary

def detect_outliers(data, max_display=10):
    """Detects outliers using the IQR method."""
    num_cols = data.select_dtypes(include=['number']).columns
    if num_cols.empty:
        print("No numerical columns found.")
        return pd.DataFrame()  # Return an empty DataFrame if no numerical columns are found

    results = []
    for col in num_cols:
        q1, q3 = data[col].quantile([0.25, 0.75])
        iqr = q3 - q1  
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr  
        outliers = data[(data[col] < low) | (data[col] > high)][col]
        if outliers.empty:
            continue

        total_distinct = data[col].nunique()
        outlier_distinct = outliers.nunique()
        outlier_percentage = round((len(outliers) / len(data[col])) * 100, 2)  

        results.append({
            'Column': col, 'Q1': round(q1, 4), 'Q3': round(q3, 4), 'IQR': round(iqr, 4),
            'Lower Bound': round(low, 4), 'Upper Bound': round(high, 4),
            'Total Distinct': total_distinct, 'Outliers Distinct': outlier_distinct,
            'Outliers Count': len(outliers), 'Outliers %': f"{outlier_percentage}%",
            'Outliers (First 10)': ", ".join(map(str, sorted(outliers.unique())[:max_display])) + 
                                   ("..." if outlier_distinct > max_display else "")
        })

    return pd.DataFrame(results) if results else pd.DataFrame() 


#=================================================================
#                  visualization    
#=================================================================


def show_missing(data):
    plt.figure(figsize=(20, 8))
    msno.matrix(data, color=(0.27, 0.50, 0.70))
    plt.title("Missing Values Matrix", fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(20, 8))
    msno.bar(data, color=sns.color_palette("Dark2", n_colors=data.shape[1]))
    plt.title("Missing Values Bar Chart", fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_boxplots(df):
    # Select only numerical columns
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if not num_cols:
        print("No numerical columns found in the dataset.")
        return
    num_features = len(num_cols)
    cols_per_row = 5  # Set the number of columns per row
    rows = int(np.ceil(num_features / cols_per_row))  
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row * 3.5, rows * 4))  # Adjusted width for better visualization
    axes = axes.flatten()  # Flatten axes array for easier indexing
    for idx, col in enumerate(num_cols):
        if idx < len(axes):
            sns.boxplot(x=df[col], ax=axes[idx])
            axes[idx].set_title(col)
            axes[idx].set_xlabel('')
            axes[idx].grid(True)
    
    # Remove any empty subplots if there are any
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

def kde_batches(data, batch_num=None):
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numerical_cols:
        print("No numerical columns found.")
        return
    
    max_subplots = 12
    total_batches = (len(numerical_cols) + max_subplots - 1) // max_subplots
    batch_mapping = {i + 1: numerical_cols[i * max_subplots: (i + 1) * max_subplots] for i in range(total_batches)}
    
    # Show available batches as a DataFrame if batch_num is not provided
    if batch_num is None:
        df_batches = pd.DataFrame(list(batch_mapping.items()), columns=["Batch Number", "Columns"])
        return df_batches
    
    # Validate batch_num
    if batch_num not in batch_mapping:
        print(f"\nBatch {batch_num} does not exist.")
        return
    
    batch_cols = batch_mapping[batch_num]
    rows, cols = (len(batch_cols) + 2) // 3, min(3, len(batch_cols))
    
    plt.figure(figsize=(cols * 7, rows * 5))
    
    for j, col in enumerate(batch_cols, 1):
        plt.subplot(rows, cols, j)
        sns.histplot(data[col], bins=20, kde=True, color='skyblue', edgecolor='black', alpha=0.7)
        
        # Add statistical lines
        for stat, (val, color) in {
            'Mean': (data[col].mean(), 'darkred'),
            'Median': (data[col].median(), 'darkgreen'),
            'Mode': (data[col].mode()[0] if not data[col].mode().empty else np.nan, 'darkblue'),
            'Min': (data[col].min(), 'darkmagenta'),
            '25%': (data[col].quantile(0.25), 'darkorange'),
            '75%': (data[col].quantile(0.75), 'darkcyan'),
            'Max': (data[col].max(), 'darkviolet')
        }.items():
            plt.axvline(val, color=color, linestyle='--', linewidth=2, label=f'{stat}: {val:.2f}')
        
        plt.title(f'{col}', fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend(loc='upper right', fontsize=10, frameon=False)
        plt.grid(False)
    
    plt.suptitle('KDE Plots', fontsize=16, fontweight='bold')
    plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.95])
    plt.show()

def box_plot_batches(data, batch_num=None):
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numerical_cols:
        print("No numerical columns found.")
        return
    
    max_subplots = 12
    total_batches = (len(numerical_cols) + max_subplots - 1) // max_subplots
    batch_mapping = {i + 1: numerical_cols[i * max_subplots: (i + 1) * max_subplots] for i in range(total_batches)}
    
    # Show available batches as a DataFrame if batch_num is not provided
    if batch_num is None:
        df_batches = pd.DataFrame(list(batch_mapping.items()), columns=["Batch Number", "Columns"])
        return df_batches
    
    # Validate batch_num
    if batch_num not in batch_mapping:
        print(f"\nBatch {batch_num} does not exist.")
        return
    
    batch_cols = batch_mapping[batch_num]
    rows, cols = (len(batch_cols) + 2) // 3, min(3, len(batch_cols))
    
    plt.figure(figsize=(cols * 7, rows * 5))
    
    for j, col in enumerate(batch_cols, 1):
        plt.subplot(rows, cols, j)
        sns.boxplot(x=data[col], color='skyblue', fliersize=5, linewidth=2)
        
        # Add statistical lines
        for stat, (val, color) in {
            'Mean': (data[col].mean(), 'darkred'),
            'Median': (data[col].median(), 'darkgreen'),
            'Min': (data[col].min(), 'darkblue'),
            '25%': (data[col].quantile(0.25), 'darkorange'),
            '75%': (data[col].quantile(0.75), 'darkcyan'),
            'Max': (data[col].max(), 'darkviolet')
        }.items():
            plt.axvline(val, color=color, linestyle='--', linewidth=2, label=f'{stat}: {val:.2f}')
        
        plt.title(f'{col}', fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.legend(loc='upper right', fontsize=10, frameon=False)
        plt.grid(False)
    
    plt.suptitle('Box Plots', fontsize=16, fontweight='bold')
    plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.95])
    plt.show()

def qq_plot_batches(data, batch_num=None):
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numerical_cols:
        print("No numerical columns found.")
        return
    
    max_subplots = 12
    total_batches = (len(numerical_cols) + max_subplots - 1) // max_subplots
    batch_mapping = {i + 1: numerical_cols[i * max_subplots: (i + 1) * max_subplots] for i in range(total_batches)}
    
    # Show available batches as a DataFrame if batch_num is not provided
    if batch_num is None:
        df_batches = pd.DataFrame(list(batch_mapping.items()), columns=["Batch Number", "Columns"])
        return df_batches
    
    # Validate batch_num
    if batch_num not in batch_mapping:
        print(f"\nBatch {batch_num} does not exist.")
        return
    
    batch_cols = batch_mapping[batch_num]
    rows, cols = (len(batch_cols) + 2) // 3, min(3, len(batch_cols))
    
    plt.figure(figsize=(cols * 7, rows * 5))
    
    for j, col in enumerate(batch_cols, 1):
        plt.subplot(rows, cols, j)
        
        # QQ Plot computation
        osm, osr = stats.probplot(data[col], dist="norm")[0]
        plt.scatter(osm, osr, s=10, color='blue', alpha=0.6, label='Data Points')
        plt.plot(osm, np.poly1d(np.polyfit(osm, osr, 1))(osm), 'r-', linewidth=2, label='Best Fit Line')
        
        plt.title(f'QQ Plot of {col}', fontsize=14)
        plt.xlabel('Theoretical Quantiles', fontsize=12)
        plt.ylabel(f'Quantiles of {col}', fontsize=12)
        plt.legend(loc='upper left', fontsize=10, frameon=False)
        plt.grid(False)
    
    plt.suptitle('QQ Plots', fontsize=16, fontweight='bold')
    plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.95])
    plt.show()

def num_vs_num_scatterplot_pair_batch(data_copy, pair_num=None, batch_num=None, hue_column=None):
    # Extract numerical columns
    numerical_cols = data_copy.select_dtypes(include=['number']).columns.tolist()
    
    if not numerical_cols:
        print("No numerical columns found in the dataset.")
        return
    
    # Generate the initial DataFrame with pair_num, pair_column, batch_num, batch_columns
    if pair_num is None and batch_num is None:
        pair_list = []
        max_subplots = 12
        
        for idx, primary_var in enumerate(numerical_cols):
            paired_vars = [var for var in numerical_cols if var != primary_var]
            num_batches = (len(paired_vars) + max_subplots - 1) // max_subplots  # Calculate the number of batches
            
            for batch_idx in range(num_batches):
                batch_pairs = paired_vars[batch_idx * max_subplots: (batch_idx + 1) * max_subplots]
                pair_list.append([idx, primary_var, batch_idx + 1, batch_pairs])
        
        df_pairs = pd.DataFrame(pair_list, columns=["Pair_Num", "Pair_Column", "Batch_Num", "Batch_Columns"])
        df_pairs = df_pairs.sort_values(by=["Pair_Num", "Batch_Num"]).reset_index(drop=True)
        
        return df_pairs  # Return the DataFrame instead of printing

    # If pair_num and batch_num are specified, generate visualizations
    if pair_num is not None and batch_num is not None:
        if pair_num < 0 or pair_num >= len(numerical_cols):
            print("Invalid pair_num. Please provide a valid numerical column index.")
            return
        
        primary_var = numerical_cols[pair_num]
        paired_vars = [var for var in numerical_cols if var != primary_var]
        max_subplots = 12
        num_batches = (len(paired_vars) + max_subplots - 1) // max_subplots
        
        if batch_num < 1 or batch_num > num_batches:
            print(f"Invalid batch_num. Please provide a value between 1 and {num_batches}.")
            return
        
        batch_pairs = paired_vars[(batch_num - 1) * max_subplots: batch_num * max_subplots]
        num_pairs = len(batch_pairs)
        cols = 3
        rows = (num_pairs // cols) + (num_pairs % cols > 0)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 5))
        axes = axes.flatten()
        
        for i, var2 in enumerate(batch_pairs):
            if hue_column and hue_column in data_copy.columns:
                sns.scatterplot(data=data_copy, x=primary_var, y=var2, hue=hue_column, palette='coolwarm', ax=axes[i])
            else:
                sns.scatterplot(data=data_copy, x=primary_var, y=var2, ax=axes[i])
            sns.regplot(data=data_copy, x=primary_var, y=var2, scatter=False, color="green", ax=axes[i], ci=None)
            axes[i].set_title(f'{primary_var} vs. {var2}')
            axes[i].set_xlabel(primary_var)
            axes[i].set_ylabel(var2)

        # Hide any remaining empty axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(f'Scatter Plots', fontsize=16, fontweight='bold')
        plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.95])
        plt.show()

def cat_vs_cat_pair_batch(data_copy, pair_num=None, batch_num=None, high_cardinality_limit=19, show_high_cardinality=True):
    # Extract categorical columns
    categorical_cols = data_copy.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_cols:
        print("No categorical columns found in the dataset.")
        return

    # Detect time series columns
    time_series_cols = [col for col in data_copy.columns if is_datetime64_any_dtype(data_copy[col])]

    # Detect high-cardinality categorical columns
    high_cardinality_cols = [col for col in categorical_cols if data_copy[col].nunique() > high_cardinality_limit]

    # Print detected high-cardinality columns
    if high_cardinality_cols:
        print(f"\n=== High Cardinality Columns (>{high_cardinality_limit} unique values) ===")
        for col in high_cardinality_cols:
            print(f"{col}: {data_copy[col].nunique()} unique values")

    # **Step 2: Handle High-Cardinality Columns Based on `show_high_cardinality`**
    modified_data = data_copy.copy()

    if show_high_cardinality:
        for col in high_cardinality_cols:
            top_categories = modified_data[col].value_counts().nlargest(high_cardinality_limit).index  # Top N categories
            modified_data[col] = modified_data[col].apply(lambda x: x if x in top_categories else 'Other')  # Group others
    else:
        # **Exclude high-cardinality columns from categorical list**
        categorical_cols = [col for col in categorical_cols if col not in high_cardinality_cols]

    # Remove excluded columns from categorical list
    filtered_categorical_cols = [col for col in categorical_cols if col not in time_series_cols]

    # **Generate DataFrame of Possible Pairs (Fix for Wrong Unique Values)**
    if pair_num is None and batch_num is None:
        pair_list = []
        max_subplots = 12  # Fixed 4x3 grid per batch
        
        for idx, primary_var in enumerate(filtered_categorical_cols):
            original_unique = data_copy[primary_var].nunique()  # **Fix: Get from original data**
            modified_unique = modified_data[primary_var].nunique()  # **Post high-cardinality grouping**

            paired_vars = [var for var in filtered_categorical_cols if var != primary_var]
            num_batches = (len(paired_vars) + max_subplots - 1) // max_subplots  # Calculate batch count
            
            for batch_idx in range(num_batches):
                batch_pairs = paired_vars[batch_idx * max_subplots: (batch_idx + 1) * max_subplots]

                pair_list.append([
                    idx, 
                    primary_var, 
                    original_unique,  # **Fix: Use original unique count**
                    modified_unique,  # **New: Store modified unique count**
                    batch_idx + 1, 
                    batch_pairs
                ])
        
        df_pairs = pd.DataFrame(pair_list, columns=[
            "Pair_Num", 
            "Pair_Column", 
            "Original_Unique",  # **Fix: Correct column name**
            "plot_Unique",  # **New: Show grouped unique count**
            "Batch_Num", 
            "Batch_Columns"
        ])
        
        df_pairs = df_pairs.sort_values(by=["Pair_Num", "Batch_Num"]).reset_index(drop=True)
        return df_pairs  # Return the DataFrame with correct unique value counts
        
    # **If pair_num and batch_num are specified, generate visualizations**
    if pair_num is not None and batch_num is not None:
        if pair_num < 0 or pair_num >= len(filtered_categorical_cols):
            print("Invalid pair_num. Please provide a valid categorical column index.")
            return
        
        primary_var = filtered_categorical_cols[pair_num]
        paired_vars = [var for var in filtered_categorical_cols if var != primary_var]
        max_subplots = 12
        num_batches = (len(paired_vars) + max_subplots - 1) // max_subplots
        
        if batch_num < 1 or batch_num > num_batches:
            print(f"Invalid batch_num. Please provide a value between 1 and {num_batches}.")
            return
        
        batch_pairs = paired_vars[(batch_num - 1) * max_subplots: batch_num * max_subplots]
        num_pairs = len(batch_pairs)

        # **Ensure at least 3 columns in layout**
        cols = 3
        rows = (num_pairs // cols) + (num_pairs % cols > 0)  # Ensures blank spaces for fewer plots

        # **Fixed Figure Size**
        fig, axes = plt.subplots(rows, cols, figsize=(24, rows * 6))  # Increased width for more space
        axes = np.ravel(axes)  # Flatten for easy iteration
        
        for i, var2 in enumerate(batch_pairs):
            contingency_table = pd.crosstab(modified_data[primary_var], modified_data[var2])

            sns.heatmap(
                contingency_table,
                annot=True,
                fmt="d",
                cmap="YlGnBu",
                ax=axes[i],
                annot_kws={"size": 8},  # Increase annotation size
                cbar=True
            )

            # **Auto-Wrap Long Titles into Two Lines**
            title_text = f'{primary_var} vs. {var2}'
            wrapped_title = "\n".join(textwrap.wrap(title_text, width=30))  # Wrap title at 30 characters

            axes[i].set_title(wrapped_title, fontsize=14, pad=20, loc='center')

            # **Fix X-Axis Label Overlapping for High-Cardinality**
            x_labels = contingency_table.columns.tolist()
            tick_interval = max(1, len(x_labels) // 20)  # Adjust to avoid overcrowding
            axes[i].set_xticks(range(len(x_labels))[::tick_interval])  # Reduce number of ticks
            axes[i].set_xticklabels(
                [x_labels[idx] for idx in range(len(x_labels))[::tick_interval]], 
                rotation=90, fontsize=10, ha='center'  # Rotate and center-align
            )

            axes[i].set_xlabel(var2, fontsize=12)
            axes[i].set_ylabel(primary_var, fontsize=12)

            # **Prevent Label Overlapping**
            axes[i].tick_params(axis='x', rotation=90, labelsize=10)
            axes[i].tick_params(axis='y', rotation=0, labelsize=10)

        # **Hide Empty Subplots (Keep 3-Column Layout)**
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")  # Keep space but make blank

        # **Improve Layout Spacing**
        fig.subplots_adjust(bottom=0.3, top=0.9, wspace=0.5, hspace=0.9)  # Increased bottom spacing
        fig.suptitle('Categorical Heatmaps', fontsize=20, x=0.5, y=1.05)  # Adjusted title position

        fig.tight_layout()  # Adjust layout dynamically
        plt.show()

def num_vs_cat_box_violin_pair_batch(data_copy, pair_num=None, batch_num=None, high_cardinality_limit=20, show_high_cardinality=True):
    # **Step 1: Detect and Print Excluded Columns**
    numerical_cols = data_copy.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = data_copy.select_dtypes(include=['object', 'category']).columns.tolist()

    if not numerical_cols or not categorical_cols:
        print("No numerical or categorical columns found.")
        return None

    # Detect time series columns
    time_series_cols = [col for col in data_copy.columns if is_datetime64_any_dtype(data_copy[col])]

    # Detect high-cardinality categorical columns
    high_cardinality_cols = [col for col in categorical_cols if data_copy[col].nunique() > high_cardinality_limit]

    # Print detected high-cardinality columns
    if high_cardinality_cols:
        print(f"\n=== High Cardinality Columns (>{high_cardinality_limit} unique values) ===")
        for col in high_cardinality_cols:
            print(f"{col}: {data_copy[col].nunique()} unique values")

    # **Step 2: Handle High-Cardinality Columns Based on `show_high_cardinality`**
    modified_data = data_copy.copy()
    
    if show_high_cardinality:
        for col in high_cardinality_cols:
            top_categories = modified_data[col].value_counts().nlargest(high_cardinality_limit).index  # Top N categories
            modified_data[col] = modified_data[col].apply(lambda x: x if x in top_categories else 'Other')  # Group others
    else:
        # **Exclude high-cardinality columns from categorical list**
        categorical_cols = [col for col in categorical_cols if col not in high_cardinality_cols]

    # Filter categorical columns (exclude time series but conditionally keep high-cardinality ones)
    filtered_categorical_cols = [col for col in categorical_cols if col not in time_series_cols]

    # **Step 3: Generate and Return DataFrame of Available Pairs**
    pair_list = []
    max_subplots = 12  # Fixed batch size

    for idx, primary_var in enumerate(numerical_cols):
        paired_vars = filtered_categorical_cols  # Each numerical column is paired with remaining categorical columns
        num_batches = (len(paired_vars) + max_subplots - 1) // max_subplots  # Calculate the number of batches

        for batch_idx in range(num_batches):
            batch_pairs = paired_vars[batch_idx * max_subplots: (batch_idx + 1) * max_subplots]

            pair_list.append([idx, primary_var, batch_idx + 1, batch_pairs])

    df_pairs = pd.DataFrame(pair_list, columns=["pair_num", "pair_column", "batch_num", "batch_column"])
    df_pairs = df_pairs.sort_values(by=["pair_num", "batch_num"]).reset_index(drop=True)

    # If no pair_num or batch_num is specified, return the DataFrame
    if pair_num is None or batch_num is None:
        return df_pairs

    # **Step 4: Validate and Plot**
    if pair_num not in df_pairs["pair_num"].unique():
        print("Invalid pair_num. Please provide a valid numerical column index.")
        return

    # Select the relevant row based on `pair_num` and `batch_num`
    selected_pair = df_pairs[(df_pairs["pair_num"] == pair_num) & (df_pairs["batch_num"] == batch_num)]

    if selected_pair.empty:
        print(f"Invalid batch_num for pair_num {pair_num}. Please check the DataFrame.")
        return

    primary_num_var = selected_pair["pair_column"].values[0]
    batch_pairs = selected_pair["batch_column"].values[0]
    num_pairs = len(batch_pairs)

    # Ensure at least 3 columns in layout
    cols = 3
    rows = (num_pairs // cols) + (num_pairs % cols > 0)  # Ensures blank spaces for fewer plots

    # **Set Fixed Figure Size**
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), constrained_layout=True)
    axes = np.ravel(axes)  # Flatten for easy iteration

    for i, cat_var in enumerate(batch_pairs):
        # **Auto-adjust x-label font size based on number of unique values**
        x_label_size = 10 if modified_data[cat_var].nunique() <= 5 else 8

        sns.boxplot(
            x=cat_var, y=primary_num_var, data=modified_data, ax=axes[i], 
            palette="Set3", boxprops=dict(alpha=0.6), width=0.4
        )

        sns.violinplot(
            x=cat_var, y=primary_num_var, data=modified_data, ax=axes[i], 
            palette="pastel", inner=None, width=0.8, alpha=0.3
        )

        # **Improve Readability: Title and Labels**
        wrapped_title = "\n".join(textwrap.wrap(f'{primary_num_var} by {cat_var}', width=30))
        axes[i].set_title(wrapped_title, fontsize=12, pad=20, loc='center')

        axes[i].tick_params(axis='x', rotation=90, labelsize=x_label_size)
        axes[i].set_xlabel(cat_var, fontsize=10)
        axes[i].set_ylabel(primary_num_var, fontsize=10)

    # **Hide Empty Subplots**
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")  # Keep space but make blank

    # **Perfectly Centered Layout Title with More Space**
    plt.suptitle('Numerical vs Categorical Box & Violin Plots', fontsize=16, x=0.5, y=1.08)

    # **Adjust spacing between title and subplots**
    plt.subplots_adjust(top=0.88)

    plt.show()

def cat_bar_batches(data, batch_num=None, high_cardinality_limit=19, show_high_cardinality=True, show_percentage=None):
    # **Set Seaborn Theme & Aesthetics**
    sns.set_theme(style="darkgrid")  # Updated theme for better contrast

    # **Step 1: Detect and Print Excluded Columns**
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_cols:
        print("No categorical columns found.")
        return None

    # Detect time series columns
    time_series_cols = [col for col in data.columns if is_datetime64_any_dtype(data[col])]

    # Detect categorical columns with > high_cardinality_limit unique values
    high_cardinality_cols = [col for col in categorical_cols if data[col].nunique() > high_cardinality_limit]

    # Print detected columns
    if time_series_cols or high_cardinality_cols:
        print("\n=== Excluded or Processed Columns ===")
        if time_series_cols:
            print(f"Time Series Columns (Auto-Detected): {time_series_cols}")
        if high_cardinality_cols:
            print(f"High Cardinality Columns (>{high_cardinality_limit} unique values): {high_cardinality_cols}")

    # **Step 2: Handle High-Cardinality Columns Based on `show_high_cardinality`**
    modified_data = data.copy()

    original_value_counts = {}  # Stores original counts for correct percentage calculations

    if show_high_cardinality:
        for col in high_cardinality_cols:
            original_value_counts[col] = data[col].value_counts(normalize=True) * 100  # Store actual percentages
            top_categories = modified_data[col].value_counts().nlargest(high_cardinality_limit).index  # Top N categories
            modified_data[col] = modified_data[col].apply(lambda x: x if x in top_categories else 'Other')  # Group others
    else:
        # **Exclude high-cardinality columns from categorical list**
        categorical_cols = [col for col in categorical_cols if col not in high_cardinality_cols]

    # Remove excluded columns from categorical list
    filtered_categorical_cols = [col for col in categorical_cols if col not in time_series_cols]

    # **Step 3: Generate and Return DataFrame of Available Batches**
    max_subplots = 12  # Fixed batch size
    total_batches = (len(filtered_categorical_cols) + max_subplots - 1) // max_subplots
    batch_mapping = {i + 1: filtered_categorical_cols[i * max_subplots: (i + 1) * max_subplots] for i in range(total_batches)}

    df_batches = pd.DataFrame(list(batch_mapping.items()), columns=["batch_num", "batch_columns"])
    
    # If batch_num is not provided, return the DataFrame
    if batch_num is None:
        return df_batches

    # **Step 4: Validate and Plot**
    if batch_num not in batch_mapping:
        print(f"\nBatch {batch_num} does not exist.")
        return

    batch_cols = batch_mapping[batch_num]
    num_pairs = len(batch_cols)

    # Ensure at least 3 columns in layout
    cols = 3
    rows = (num_pairs // cols) + (num_pairs % cols > 0)  # Ensures blank spaces for fewer plots

    # **Increased Figure Size for Better Spacing**
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7.5, rows * 5.5), constrained_layout=True)
    axes = np.ravel(axes)  # Flatten for easy iteration

    for j, col in enumerate(batch_cols):
        ax = axes[j]

        # **Plot the bar chart with Updated Color Palette**
        value_counts = modified_data[col].value_counts()
        total_count = data[col].value_counts().sum()  # Get total count from original dataset

        bar_plot = sns.barplot(
            x=value_counts.index, 
            y=value_counts.values, 
            ax=ax, palette="coolwarm", edgecolor="black"
        )

        # **Ensure values are completely above the bars**
        ylim = ax.get_ylim()
        max_height = ylim[1] * 1.15  # Increased space above the bars
        ax.set_ylim(0, max_height)  # Update plot limits

        # **Fix Label Merging Issue & Improve Readability**
        rotation_angle = 50  # Always rotate labels at 50 degrees
        fontsize = 10  # Fixed font size for labels

        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation_angle, ha="right", fontsize=fontsize)

        for p in bar_plot.patches:
            height = p.get_height()
            if height > 0:
                # **Correct Percentage Calculation Using Original Data**
                if col in original_value_counts:
                    percentage = original_value_counts[col].get(p.get_x(), 0)
                else:
                    percentage = (height / total_count) * 100  # Compute directly if not high-cardinality

                # **Conditionally display either percentage, count, or both**
                if show_percentage:
                    label_text = f"{percentage:.1f}%"
                else:
                    label_text = f"{int(height)}"
                
                label_position = height * 1.07  # Ensures values are fully above the bars
                ax.annotate(label_text, 
                            (p.get_x() + p.get_width() / 2., label_position),  
                            ha="center", va="bottom", fontsize=fontsize, color="black")

        # **Auto-wrap long subplot titles**
        wrapped_title = "\n".join(textwrap.wrap(f'Distribution of {col}', width=35))
        ax.set_title(wrapped_title, fontsize=14, pad=30, loc='center')

        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel("Count", fontsize=12)

    # **Hide Empty Subplots**
    for k in range(len(batch_cols), len(axes)):
        axes[k].axis("off")  # Keep space but make blank

    # **Perfectly Centered Layout Title with More Space**
    plt.suptitle('Categorical Bar Plots', fontsize=18, x=0.5, y=1.08, fontweight='bold')

    # **Adjusts spacing between title and subplots**
    plt.subplots_adjust(top=0.88)

    plt.show()

def cat_pie_chart_batches(data, batch_num=None, high_cardinality_limit=20):
    # **Step 1: Detect and Print Excluded Columns**
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_cols:
        print("No categorical columns found.")
        return None

    # Detect time series columns
    time_series_cols = [col for col in data.columns if is_datetime64_any_dtype(data[col])]

    # Detect categorical columns with more unique values than the limit
    high_cardinality_cols = [col for col in categorical_cols if data[col].nunique() > high_cardinality_limit]

    # Print detected columns (excluded from analysis)
    if time_series_cols or high_cardinality_cols:
        print("\n=== Excluded Columns ===")
        if time_series_cols:
            print(f"Time Series Columns (Auto-Detected): {time_series_cols}")
        if high_cardinality_cols:
            print(f"High Cardinality Columns (> {high_cardinality_limit} unique values): {high_cardinality_cols}")

    # Remove excluded columns from categorical list
    filtered_categorical_cols = [col for col in categorical_cols if col not in time_series_cols + high_cardinality_cols]

    # **Step 2: Generate and Return DataFrame of Available Batches**
    max_subplots = 12  # Fixed batch size
    total_batches = (len(filtered_categorical_cols) + max_subplots - 1) // max_subplots
    batch_mapping = {i + 1: filtered_categorical_cols[i * max_subplots: (i + 1) * max_subplots] for i in range(total_batches)}

    df_batches = pd.DataFrame(list(batch_mapping.items()), columns=["batch_num", "batch_columns"])
    
    # If batch_num is not provided, return the DataFrame
    if batch_num is None:
        return df_batches

    # **Step 3: Validate and Plot**
    if batch_num not in batch_mapping:
        print(f"\nBatch {batch_num} does not exist.")
        return

    batch_cols = batch_mapping[batch_num]
    num_pairs = len(batch_cols)

    # Ensure at least 3 columns in layout
    cols = 3
    rows = (num_pairs // cols) + (num_pairs % cols > 0)  # Ensures blank spaces for fewer plots

    # **Increase Figure Size for Visibility**
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 6), constrained_layout=True)
    axes = np.ravel(axes)  # Flatten for easy iteration

    for j, col in enumerate(batch_cols):
        ax = axes[j]

        # **Plot the Pie Chart**
        series = data[col].value_counts()
        sizes = series.values / series.sum() * 100  # Convert to percentages
        colors = plt.cm.Paired(np.linspace(0, 1, len(series)))  # High-contrast colors

        wedges, texts, autotexts = ax.pie(
            sizes, autopct=lambda p: f'{p:.1f}%' if p > 2 else '', startangle=90, 
            colors=colors, pctdistance=0.8, textprops={'fontsize': 12}
        )

        # **Ensure category labels do not overlap**
        for text in texts:
            text.set_fontsize(11)

        # **Move Small Percentage Labels Outside for Readability**
        for text in autotexts:
            text.set_fontsize(12)

        # **Title Formatting**
        ax.set_title(f'Distribution of {col}', fontsize=14, pad=25, loc='center')

        # **Legend Placement Outside the Pie Chart**
        legend_labels = [f'{label} ({size:.1f}%)' for label, size in zip(series.index, sizes)]
        ax.legend(
            wedges, legend_labels, title=col, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11, frameon=False
        )

    # **Hide Empty Subplots**
    for k in range(len(batch_cols), len(axes)):
        axes[k].axis("off")  # Keep space but make blank

    # **Perfectly Centered Layout Title with More Space**
    plt.suptitle('Pie Charts of Categorical Variables', fontsize=18, x=0.5, y=1.08, fontweight='bold')

    # **Adjust spacing between title and subplots**
    plt.subplots_adjust(top=0.85)

def num_analysis_and_plot(data, attr, target=None, visualize=True, subplot=True, show_table=True, target_vis=True, return_df=None):
    """Analyze numerical attributes with/without a target variable and visualize them."""

    if attr not in data.columns:
        print(f"Attribute '{attr}' not found.")
        return

    # Compute overall statistics
    summary = pd.DataFrame(calc_stats(data[attr]), index=['Overall']).T

    if target and target in data.columns:
        grouped_stats = {grp: calc_stats(grp_data) for grp, grp_data in data.groupby(target)[attr]}
        for grp, stats in grouped_stats.items():
            summary = summary.join(pd.DataFrame(stats, index=[f"{target}: {grp}"]).T)

    # Reset index to ensure no duplicate columns
    summary = summary.reset_index()

    # Move attribute name to the first column
    summary.insert(0, attr.title(), summary.pop("index"))

    if show_table:
        print(f"\n### Analysis for '{attr}' {'by ' + target if target else ''} ###\n")
        print(tabulate(summary, headers="keys", tablefmt="pipe"))

    # Visualization
    if visualize:
        if subplot:
            fig, axes = plt.subplots(1, 2 if target_vis else 1, figsize=(18, 6))

            # Histogram with KDE
            sns.histplot(data, x=attr, hue=target, bins=30, kde=True, palette='Set1', ax=axes[0])
            axes[0].set_title(f'Histogram of {attr}' + (f' by {target}' if target else ''))
            axes[0].tick_params(axis='x', rotation=50)  # Rotate X-axis labels

            # Box plot (only if target_vis=True)
            if target_vis:
                if target:
                    sns.boxplot(x=target, y=attr, data=data, palette='Set2', ax=axes[1])
                    axes[1].set_title(f'Box Plot of {attr} by {target}')
                    axes[1].tick_params(axis='x', rotation=50)  # Rotate X-axis labels

                    # Adjust legend position if too many target values
                    if len(data[target].unique()) > 10:
                        axes[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
                else:
                    sns.boxplot(y=data[attr], palette='Set2', ax=axes[1])
                    axes[1].set_title(f'Box Plot of {attr}')

            plt.tight_layout()
            plt.show()
        else:
            # Separate plots for large datasets
            plt.figure(figsize=(12, 6))
            sns.histplot(data, x=attr, hue=target, bins=30, kde=True, palette='Set1')
            plt.title(f'Histogram of {attr}' + (f' by {target}' if target else ''))
            plt.xticks(rotation=50)  # Rotate X-axis labels
            plt.show()

            if target_vis:
                plt.figure(figsize=(8, 6))
                if target:
                    sns.boxplot(x=target, y=attr, data=data, palette='Set2')
                    plt.title(f'Box Plot of {attr} by {target}')
                    plt.xticks(rotation=50)  # Rotate X-axis labels

                    if len(data[target].unique()) > 10:
                        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                else:
                    sns.boxplot(y=data[attr], palette='Set2')
                    plt.title(f'Box Plot of {attr}')
                plt.show()

    # Return table only if explicitly specified
    if return_df is not None:
        return summary

def cat_analyze_and_plot(data, attribute, target=None, visualize=True, target_vis=True, show_table=True, subplot=True, return_df=None):
    """Analyze categorical attribute with/without a target and visualize it with better label spacing."""

    # Compute value counts and percentages
    value_counts = data[attribute].value_counts().to_frame(name='Count')
    percentages = (value_counts / value_counts.sum() * 100).round(2).rename(columns={'Count': '% Total'})

    # Merge and format table
    final_table = pd.concat([value_counts, percentages], axis=1)

    if target and target in data.columns:
        grouped_counts = data.groupby([attribute, target]).size().unstack(fill_value=0)
        grouped_percentages = grouped_counts.div(grouped_counts.sum(axis=1), axis=0).mul(100).round(2)
        grouped_percentages.columns = [f'% {col}' for col in grouped_percentages.columns]
        final_table = pd.concat([grouped_counts, final_table, grouped_percentages], axis=1)

    # Reset index to ensure attribute column is not duplicated
    final_table = final_table.reset_index()

    # Move the attribute column to the first position
    final_table.insert(0, attribute.title(), final_table.pop(attribute))

    # Sort table by count in descending order
    final_table = final_table.sort_values(by='Count', ascending=False)

    if show_table:
        # Print table (sorted)
        print(f"\nValue counts and percentages for {attribute.title()}" + (f" and {target.title()}:\n" if target else ":\n"))
        print(final_table.to_markdown(index=False, tablefmt="pipe"))

    # Visualization
    if visualize:
        sorted_data = value_counts.index  # Sorted order

        # Adjust figure size based on category count
        fig_width = max(12, len(sorted_data) * 0.4)  # Dynamic width
        fig_height = 6 + (len(sorted_data) * 0.02)   # Increase height for more categories
        
        if subplot and (target and target_vis):
            # Create side-by-side subplots
            fig, axes = plt.subplots(1, 2, figsize=(fig_width * 1.5, fig_height))
            
            # General Distribution
            sns.barplot(data=value_counts.reset_index(), x=attribute, y='Count', order=sorted_data, palette="pastel", ax=axes[0])
            axes[0].set_title(f'{attribute.title()} Distribution')
            axes[0].tick_params(axis='x', rotation=45, labelsize=10)
            
            # Target-based Distribution
            sns.countplot(data=data, x=attribute, hue=target, order=sorted_data, palette="pastel", ax=axes[1])
            axes[1].set_title(f'{attribute.title()} by {target.title()}')
            axes[1].tick_params(axis='x', rotation=45, labelsize=10)

            # Annotate both plots
            for ax in axes:
                for p in ax.patches:
                    if p.get_height() > 0:
                        annotation_y = p.get_height() + (p.get_height() * 0.02)
                        ax.annotate(f'{int(p.get_height())}', 
                                    (p.get_x() + p.get_width() / 2., annotation_y), 
                                    ha='center', va='bottom', fontsize=9, color='black', 
                                    xytext=(0, 3), textcoords='offset points')

            plt.tight_layout()
            plt.show()
        
        else:
            # Separate plots
            plt.figure(figsize=(fig_width, fig_height))
            sns.barplot(data=value_counts.reset_index(), x=attribute, y='Count', order=sorted_data, palette="pastel")
            plt.title(f'{attribute.title()} Distribution')
            plt.xticks(rotation=45, ha='right', fontsize=10)

            # Annotate bars
            ax = plt.gca()
            for p in ax.patches:
                if p.get_height() > 0:
                    annotation_y = p.get_height() + (p.get_height() * 0.02)
                    ax.annotate(f'{int(p.get_height())}', 
                                (p.get_x() + p.get_width() / 2., annotation_y), 
                                ha='center', va='bottom', fontsize=9, color='black', 
                                xytext=(0, 3), textcoords='offset points')

            plt.tight_layout()
            plt.show()

            # Target-based distribution (if required)
            if target and target_vis:
                plt.figure(figsize=(fig_width, fig_height))
                sns.countplot(data=data, x=attribute, hue=target, order=sorted_data, palette="pastel")
                plt.title(f'{attribute.title()} by {target.title()}')
                plt.xticks(rotation=45, ha='right', fontsize=10)

                # Annotate bars
                ax = plt.gca()
                for p in ax.patches:
                    if p.get_height() > 0:
                        annotation_y = p.get_height() + (p.get_height() * 0.02)
                        ax.annotate(f'{int(p.get_height())}', 
                                    (p.get_x() + p.get_width() / 2., annotation_y), 
                                    ha='center', va='bottom', fontsize=9, color='black', 
                                    xytext=(0, 3), textcoords='offset points')

                plt.tight_layout()
                plt.show()

    # Return table only if explicitly specified
    if return_df is not None:
        return final_table

#=================================================================
#                           END
#=================================================================