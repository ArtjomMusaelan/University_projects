#!/usr/bin/env python
# coding: utf-8

# In[1]:


import psycopg2
import pandas as pd
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# In[2]:


def import_df(name, columns_to_convert=[]):

    df = pd.read_csv(f"./data/{name}.csv")

    for col in columns_to_convert:
        df[col] = pd.to_datetime(df[col])
    return df


# In[3]:


def import_weather_df(table_name):
    df = import_df(table_name)
    df["dtg"] = pd.to_datetime(df["dtg"])
    df = df.set_index("dtg").loc["2018-01-01":, :]
    df = df.sort_index()

    return df


# In[4]:


safe_driving_df = import_df(
    "safe_driving", columns_to_convert=["event_start", "event_end"]
)


# In[5]:


safe_driving_df.dtypes


# In[6]:


safe_driving_df.head(10)


# In[7]:


safe_driving_df["road_name"].value_counts()


# In[8]:


wind_df = import_weather_df("wind")


# In[ ]:


# In[9]:


temp_df = import_weather_df("temperature")


# In[10]:


prec_df = import_weather_df("precipitation")


# In[11]:


accidents_17_23_df = import_df("accident_data_17_23")


# In[12]:


safe_driving_df.head(10)


# We will ensure categorical values do not have uneccesary whitespace or other unexpected special characters, moreover, we will unifiy value casing to lower
#

# In[13]:


def clean_categorical_data(df):
    string_cols = [col for col in df.columns if "object" == str(df[col].dtype)]

    for col in string_cols:
        df[col] = df[col].str.strip()
        df[col] = df[col].str.replace(r"[^\w\s]", "", regex=True)
        df[col] = df[col].str.lower()

    for col in string_cols:
        df = df.rename(columns={col: str(col).lower().replace(" ", "_")})

    return df


# In[14]:


safe_driving_df = clean_categorical_data(safe_driving_df)


# In[15]:


def delete_empty_columns(df):
    columns_cols_to_drop = []

    for col in df.columns:
        if df[col].isna().sum() == df.shape[0]:
            columns_cols_to_drop.append(col)

    return df.drop(columns=columns_cols_to_drop)


# In[16]:


safe_driving_df = delete_empty_columns(safe_driving_df)


# In[17]:


#### checking missing values in dataset


def print_line_break():
    print("=" * 20)
    print(" " + "-" * 18 + " ")
    print("=" * 20)


def show_dataframe_general_info(df):
    display("General info of df")
    display(df.info())
    display("Description of df")
    display(df.describe())

    check_df_missing_values(df)


def check_df_missing_values(df):
    total_missing_values = df.isna().sum().sum()
    display(f"Total number of missing values: ", total_missing_values)

    if total_missing_values > 0:

        display("Number of missing values in particular columns: ")
        display(df.isna().sum())


def show_value_counts(df, col):
    display(f"Value counts of {col}")
    sorted_val_df = df[col].value_counts().sort_values(ascending=False)

    if sorted_val_df.shape[0] > 6:
        sorted_val_df = sorted_val_df.iloc[:6]
    display(sorted_val_df)

    print_line_break()

    display(f"Least used values in {col} column: ")

    display(df[col].value_counts().sort_values(ascending=True).iloc[:5])

    unique_vals_in_col = len(pd.unique(df[col]))

    col_dtype = str(df[col].dtype)

    if col_dtype.startswith("int") or col_dtype.startswith("float"):
        fig, ax = plt.subplots(figsize=(14, 8))

        sns.boxplot(x=col, data=df, ax=ax)
        plt.show()
    elif unique_vals_in_col < 20 and col_dtype.startswith("object"):

        fig, ax = plt.subplots(figsize=(18, 8))

        missing_vals = df[col].isna().sum()

        if missing_vals > 0:
            ax.axhline(
                y=missing_vals,
                color="r",
                linestyle="--",
                linewidth=2,
                label="Missing values in df",
            )
            ax.legend()
            sns.countplot(x=col, data=df.replace({np.nan: "unknown"}), ax=ax)
        sns.countplot(x=col, data=df, ax=ax)

        plt.show()


def show_dataframe_column_value_counts(df):
    cols = df.columns

    for col in cols:
        print_line_break()

        show_value_counts(df, col)
        missing_vals_in_col = df[col].isna().sum()

        if missing_vals_in_col > 0:
            display(f"Missing values in {col}")
            display(f"{col}: {missing_vals_in_col}")


# In[18]:


show_dataframe_general_info(safe_driving_df)


# In[19]:


show_dataframe_column_value_counts(safe_driving_df)


# In[20]:


def show_duplicated_values_in_column(df, col_name):
    display("Show duplicated values in column: {col_name}")
    total_duplicated_values = df[col_name].duplicated().sum()
    if total_duplicated_values > 0:

        display(f"Duplicated values in {col_name} :")
        display("Number of duplicated values / all rows")
        duplicated_values_perc = round(
            (total_duplicated_values / df[col_name].shape[0] * 100), 2
        )

        display(
            f"{total_duplicated_values}/{df[col_name].shape[0]} :  which is around {duplicated_values_perc}%"
        )
        sorted_val_df = df[col_name].value_counts().sort_values(ascending=False)
        sorted_val_df = sorted_val_df[sorted_val_df > 1]

        display(sorted_val_df)

        duplicated_values = sorted_val_df.reset_index()[col_name]

        display("Show duplicated column rows :")
        display(df[df[col_name].isin(duplicated_values.to_list())])
    else:
        display("No duplicated values in this column !!!")


# In[21]:


def show_general_duplicate_values(df, col_name=None):

    if col_name is not None:

        show_duplicated_values_in_column(df, col_name)
    else:
        total_duplicated_values = df.duplicated().sum()
        if total_duplicated_values > 0:

            display(f"Duplicated values in df:")
            display("Number of duplicated values / all rows")
            duplicated_values_perc = round(
                (total_duplicated_values / df.shape[0] * 100), 2
            )

            display(
                f"{total_duplicated_values}/{df.shape[0]} :  which is around {duplicated_values_perc}%"
            )
        else:
            display("No duplicated values in this dataframe !!!")


def drop_duplicates_in_df(df, columns):

    drop_duplicated = False

    if len(columns) > 1:
        for col in columns:
            display("Duplicated values in {columns} after dropping them")
            print_line_break()
            drop_duplicated = df.drop_duplicates(subset=[col], inplace=True)
            show_duplicated_values_in_column(safe_driving_df, col)

            print_line_break()
    else:
        display("Duplicated values in {columns} after dropping them")
        print_line_break()
        drop_duplicated = df.drop_duplicates(subset=[*columns], inplace=True)
        show_duplicated_values_in_column(safe_driving_df, columns[0])

    return drop_duplicated


# In[22]:


show_general_duplicate_values(safe_driving_df)


# In[23]:


show_duplicated_values_in_column(safe_driving_df, "eventid")


# In[24]:


show_duplicated_values_in_column(safe_driving_df, "event_start")


# Since most of the duplicated id constitute the similar or the same accidents and the fraction of duplicated values is relatively insignificant, the duplicated rows will be dropped

# In[25]:


drop_duplicates_in_df(safe_driving_df, ["eventid", "event_start"])


# Now I will proceed to examine outliers
#

# In[26]:


safe_driving_df.dtypes


# In[27]:


def plot_columns(df, columns, plot):
    if len(columns) == 0:
        display("No columns to plot")
        print_line_break()
        print_line_break()
        return

    cols_length = len(columns)

    fig, axes = plt.subplots(
        nrows=cols_length,
        ncols=1,
        figsize=(12, cols_length * 6),
        sharex=False,
        sharey=False,
    )
    print(axes, type(axes))

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for idx, current_ax in enumerate(axes.flatten()):
        if idx < len(columns):

            current_col = columns[idx]

            current_ax.set_title(f"Column: {current_col}")

            plot(x=df[current_col], ax=current_ax)

    plt.show()


def plot_numeric_columns(df, columns):
    plot_columns(df, columns, sns.boxplot)


def plot_string_columns(df, columns):
    plot_columns(df, columns, sns.countplot)


def plot_bool_columns(df, columns):
    plot_columns(df, columns, sns.countplot)


def plot_value_distributions_in_df(df, columns_to_avoid=[]):

    numeric_cols = [
        col
        for col in df.columns
        if ("float" in str(df[col].dtype) or "int" in str(df[col].dtype))
        and col not in columns_to_avoid
    ]

    string_cols = [
        col
        for col in df.columns
        if "object" == str(df[col].dtype) and col not in columns_to_avoid
    ]

    bool_cols = [
        col
        for col in df.columns
        if "bool" == str(df[col].dtype) and col not in columns_to_avoid
    ]

    if numeric_cols:

        display("Numerical columns plotted :")
        plot_numeric_columns(df, numeric_cols)
        print_line_break()
        print_line_break()
        print_line_break()

    if string_cols:

        display("String columns plotted :")
        plot_string_columns(df, string_cols)

        print_line_break()
        print_line_break()
        print_line_break()

    if bool_cols:

        display("Bool columns plotted :")
        plot_bool_columns(df, bool_cols)


# In[28]:


plot_value_distributions_in_df(
    safe_driving_df, ["eventid", "road_segment_id", "latitude", "longitude"]
)


# In[29]:


def show_outliers_fraction(df, col, Q1, Q3, IQR):
    print_line_break()
    display(f"The fraction of outliers in {col}")
    total_outliers_number_in_col_mask = (df[col] < Q1 - 1.5 * IQR) | (
        df[col] > Q3 + 1.5 * IQR
    )
    total_outliers_number_in_col = df[total_outliers_number_in_col_mask].shape[0]
    if total_outliers_number_in_col <= 0:
        display(f"No outliers detected in {col} column")
        return

    print(total_outliers_number_in_col)
    total_outliers_number_in_col_perc = (
        round((total_outliers_number_in_col / df.shape[0]), 2) * 100
    )
    display(
        f"{total_outliers_number_in_col}  / {df.shape[0]} which is around {total_outliers_number_in_col_perc}%"
    )
    print_line_break()


def delete_outliers(df, columns, multiplier=1.5):
    df_no_outliers = df.copy()

    for col in columns:
        Q1 = df_no_outliers[col].quantile(0.25)
        Q3 = df_no_outliers[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - (IQR * multiplier)
        upper_bound = Q3 + (IQR * multiplier)

        show_outliers_fraction(df, col, Q1, Q3, IQR)
        print(
            f"{col}: Q1={Q1}, Q3={Q3}, IQR={IQR}, Lower Bound={lower_bound}, Upper Bound={upper_bound}"
        )

        df_no_outliers = df_no_outliers[
            (df_no_outliers[col] >= lower_bound) & (df_no_outliers[col] <= upper_bound)
        ]

    return df_no_outliers


# Due to low amount of outliers located in dataset they will be removed using IQR and quantiles lower 0.25 and greater than 0.75

# In[30]:


safe_driving_df = delete_outliers(
    safe_driving_df, ["end_speed_kmh", "speed_kmh", "duration_seconds"]
)


# In[31]:


plot_value_distributions_in_df(
    safe_driving_df, ["eventid", "road_segment_id", "latitude", "longitude"]
)


# is_valid , road_manager_type, road_number,road_manager_name, municipality_name columns does not provide much value therefore they will be dropped
#

# In[32]:


def drop_columns_in_df(df, columns_to_drop):
    cols_drop_len = len(columns_to_drop)

    for col_to_drop in columns_to_drop:
        if col_to_drop in df.columns:

            df.drop(columns=[col_to_drop], inplace=True)


# In[33]:


drop_columns_in_df(
    safe_driving_df,
    [
        "is_valid",
        "road_manager_type",
        "road_number",
        "road_manager_name",
        "municipality_name",
        "place_name",
    ],
)


# ### Let's simplify incident_severity column

# In[34]:


def convert_column_to_binary(df, columns_with_new_values):
    for key, val in columns_with_new_values.items():
        col = key
        multiple_values = val["top_values"]

        new_replace_value = val["new_value"]

        most_frequent_values = df[col].value_counts().index[0:multiple_values]

        df[col] = df[col].apply(
            lambda row: row if str(row) in most_frequent_values else new_replace_value
        )


# In[35]:


columns_with_new_values_dict = {
    "incident_severity": {"new_value": "other incident severities", "top_values": 2},
}

convert_column_to_binary(safe_driving_df, columns_with_new_values_dict)


# In[36]:


plot_value_distributions_in_df(
    safe_driving_df, ["eventid", "road_segment_id", "latitude", "longitude"]
)


# Let's proceed with data inconsistencies

# In[37]:


safe_driving_df["incident_severity"].value_counts()


# In[38]:


def clip_numerical_cols(df, columns):
    for col in columns:
        df[col] = df[col].round(2)


def clean_numerical_cols(df):
    numeric_cols = [
        col
        for col in df.columns
        if ("float" in str(df[col].dtype) or "int" in str(df[col].dtype))
    ]

    for col in numeric_cols:
        df[col] = df[col].abs()
        df[col] = df[col].astype(float)


# In[39]:


clean_numerical_cols(safe_driving_df)


# In[40]:


clip_numerical_cols(safe_driving_df, ["speed_kmh", "end_speed_kmh", "maxwaarde"])


# There were some cases where initial speed_kmh was 0, we will analyze that
#
#

# In[41]:


safe_driving_df[safe_driving_df["speed_kmh"] == 0.0]


# These cases have reasonable explanation caused by Accelerating therefore these rows will not be removed

# In[42]:


safe_driving_df.describe()


# Now we will prooced with scaling data using Standard Scaler from Sklearn
#

# In[43]:


def scale_numerical_data(df, columns):

    for col in columns:
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(df[[col]])


# In[44]:


scale_numerical_data(
    safe_driving_df, ["duration_seconds", "speed_kmh", "end_speed_kmh", "maxwaarde"]
)


# In[45]:


safe_driving_df.head(10)


# Categorical encoding will be left for modelling process therefore no decoding functions will be implemented currently
#

# Lets import weather informations
#

# No we will check basic column data distribution of ff sensor 10
#

# In[46]:


wind_df["ff_sensor_10"].value_counts()


# In[47]:


sns.boxplot(x=wind_df["ff_sensor_10"])


# In[48]:


import pandas as pd
import numpy as np


def calculate_rolling_average(weather_df, value_column):

    # Calculate the rolling average with a 1-hour window, closed='both' includes both ends
    weather_df["rolling_mean"] = (
        weather_df[value_column].rolling("1h", closed="both").mean()
    )

    # Reset index for merging later
    return weather_df.reset_index()


def merge_driving_with_weather_df(
    driving_df_original, weather_df_original, value_column, new_value_column_name
):
    # Calculate the rolling average
    driving_df = driving_df_original.copy()
    weather_df = weather_df_original.copy()

    weather_df = calculate_rolling_average(weather_df, value_column)

    # Convert event_start_timestamp to datetime if not already
    driving_df["event_start"] = pd.to_datetime(driving_df["event_start"])

    # Sort both dataframes by 'dtg' for merge_asof
    weather_df = weather_df.sort_values("dtg")
    driving_df = driving_df.sort_values("event_start")

    # Use merge_asof to match each accident event with the nearest weather data point within a 1-hour window
    print(driving_df.columns)
    merged_df = pd.merge_asof(
        driving_df,
        weather_df[["dtg", "rolling_mean"]],
        left_on="event_start",
        right_on="dtg",
        direction="backward",
        tolerance=pd.Timedelta(hours=1),
    )
    print(merged_df.columns)
    # Rename the rolling mean column to the desired new_value_column_name
    merged_df = merged_df.rename(columns={"rolling_mean": new_value_column_name})
    print("After:", merged_df.columns)
    # Drop the 'dtg' column from the weather dataframe to avoid confusion
    merged_df = merged_df.drop(columns=["dtg"])

    return merged_df


# In[49]:


safe_driving_df = merge_driving_with_weather_df(
    safe_driving_df, wind_df, "ff_sensor_10", "last_hour_wind_avg"
)


# In[50]:


safe_driving_df.loc[:, ["event_start", "last_hour_wind_avg"]].describe()


# In[51]:


safe_driving_df = merge_driving_with_weather_df(
    safe_driving_df, temp_df, "t_dryb_10", "last_hour_temp_avg"
)


# In[52]:


safe_driving_df = merge_driving_with_weather_df(
    safe_driving_df, prec_df, "ri_pws_10", "last_hour_rain_avg"
)


# In[53]:


safe_driving_df.head(10)


# In[54]:


def scale_numerical_data(df, columns):

    for col in columns:
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(df[[col]])


# In[55]:


scale_numerical_data(
    safe_driving_df, ["last_hour_wind_avg", "last_hour_temp_avg", "last_hour_rain_avg"]
)


# In[56]:


safe_driving_df.head(10)


# In[57]:


safe_driving_df.shape


# ### Manage columns and transform them
#

# In[58]:


def transform_numerical_column_to_str(df, columns):
    df.loc[:, columns] = df.loc[:, columns].astype(str)


# ### We will transform Year column to categorical column in order to make it easier to plot for now

# In[59]:


transform_numerical_column_to_str(accidents_17_23_df, ["Year"])


# ### We will ensure categorical values do not have uneccesary whitespace or other unexpected special characters, moreover, we will unifiy value casing to lower
#

# In[60]:


accidents_17_23_df = clean_categorical_data(accidents_17_23_df)


# ### We will ensure that all unknown values are converted as nan values

# In[61]:


def convert_unknown_to_nan(df):
    df.replace({"unknown": np.nan}, inplace=True)


# In[62]:


convert_unknown_to_nan(accidents_17_23_df)


# ### In some columns  are presents empty '' values which we will convert to  nan values
#
#

# In[63]:


def convert_empty_values_to_nan(df, columns):
    for col in columns:
        df[col] = df[col].replace({"": np.nan})


# In[64]:


convert_empty_values_to_nan(accidents_17_23_df, ["first_mode_of_transport"])


# ### We will ensure that rows with insignificant rows will be dropped

# In[65]:


def drop_rows_with_drop_values(df, col, drop_values):

    if drop_values:
        mask = df[col].apply(lambda row: str(row) in drop_values)

        print(pd.unique(mask))

        idxs_to_drop = df[mask].index
        print(idxs_to_drop)
        df.drop(index=idxs_to_drop, inplace=True)


# In[66]:


def convert_string_column_to_numerical(df, col, drop_values=[]):
    drop_rows_with_drop_values(df, col, drop_values)

    def return_speed(row):

        splitted_row = str(row).split(" ")
        return float(splitted_row[0])

    df[col] = df[col].apply(
        lambda row: return_speed(row) if not pd.isnull(row) else row
    )


# ### Delete footpace homezone value from speed_limit column because it only occurs 6 times in whole df

# In[67]:


convert_string_column_to_numerical(
    accidents_17_23_df, "speed_limit", drop_values=["footpace  homezone"]
)


# ### Drop municipality column because it has only "breda" value

# In[68]:


drop_columns_in_df(accidents_17_23_df, ["municipality"])


# ### Show the columns with missing values

# In[69]:


def show_columns_with_missing_values(df):
    df_cols = df.columns

    for col in df_cols:
        missing_vals_in_col = df[col].isna().sum()
        if missing_vals_in_col > 0:
            nan_perc = round((missing_vals_in_col / df.shape[0]) * 100, 2)
            print(f"Col: {col} has {missing_vals_in_col} missing values")
            print(f"Percentage of missing values / all values in column: {nan_perc } %")
            show_dataframe_column_value_counts(df[[col]])
            print_line_break()


# In[70]:


show_columns_with_missing_values(accidents_17_23_df)


# ### Since every column has missing values over 15% that means that missing values constitute significant amount of important information , but because of the fact that I am not convinced how the datasets will be merged I will leave missing data imputation steps for later
#

# In[71]:


show_general_duplicate_values(accidents_17_23_df)


# ### The brief analysis of dataset indicates no explicit unique identifier for each event in dataframe. Moreover, the amount of whole rows  duplicated is 0  therefore it is safe to assume that there are not any duplicates
#

# ### Now I will proceed to examine distribution

# In[72]:


plot_value_distributions_in_df(accidents_17_23_df, columns_to_avoid=[])


# ### Due to unequal distribution of certain columns let's convert them into binary column type

# In[73]:


columns_with_new_values_dict = {
    "accident_severity": {"new_value": "injury or fatal", "top_values": 1},
    "town": {"new_value": "other city", "top_values": 1},
    "first_mode_of_transport": {"new_value": "other", "top_values": 1},
    "second_mode_of_transport": {"new_value": "other", "top_values": 2},
    "light_condition": {"new_value": "darkness or twilight", "top_values": 1},
    "road_condition": {"new_value": "wetdamp or snowblack ice", "top_values": 1},
    "road_situation": {"new_value": "other road situation", "top_values": 4},
    "weather": {"new_value": "other weather situation", "top_values": 2},
}


# In[74]:


convert_column_to_binary(accidents_17_23_df, columns_with_new_values_dict)


# Let's eleminate outliers from speed_limit by removing accidents on road with very high or very low speed limit
#
#

# In[75]:


def show_dist_for_cols(df, cols, boxplot=False):
    fig, axes = plt.subplots(nrows=len(cols), ncols=1, figsize=(20, 15))
    if len(cols) == 1:
        axes = np.array([axes])
    for idx, ax in enumerate(axes.flatten()):

        if boxplot:
            sns.boxplot(x=cols[idx], ax=ax, data=df)
        else:

            sns.countplot(x=cols[idx], ax=ax, data=df)

    plt.show()


# In[76]:


show_dist_for_cols(accidents_17_23_df, ["speed_limit", "accidents"], True)


# In[77]:


accidents_17_23_df = delete_outliers(accidents_17_23_df, ["speed_limit", "accidents"])


# In[78]:


show_dist_for_cols(accidents_17_23_df, ["speed_limit", "accidents"])


# ### Let's again present data distributions after transformations

# In[79]:


plot_value_distributions_in_df(accidents_17_23_df, columns_to_avoid=[])


# ### Let's proceed with data inconsistencies

# In[80]:


clean_numerical_cols(accidents_17_23_df)


# In[81]:


accidents_17_23_df.head(10)


# ### Let's see value counts of accidents
#
#

# In[82]:


accidents_17_23_df["accidents"].value_counts()


# ### Let's scale numerical values
#

# In[83]:


scale_numerical_data(
    accidents_17_23_df, accidents_17_23_df.select_dtypes(include=["float", "int"])
)


# In[84]:


accidents_17_23_df.describe()


# ## Given the fact that accidents number has std = 0 and now only contains value 0 after transofrmations, no longer it constains meaningful info, therefore let's drop accidents columns
#

# In[85]:


drop_columns_in_df(accidents_17_23_df, ["accidents"])


# In[86]:


accidents_17_23_df.describe()


# ### Categorical encoding will be left for modelling process therefore no decoding functions will be implemented currently

# ## Now the data needs to be analyzed

# ### Let's make a weighted mean of accident_severity table to make it a new column for safe_driving_df
# So in our case the weighted mean of types of accidents severity will help us to assess if the street is high or low risk

# In[87]:


def transform_acc_sev_col_to_encoding(df):
    df = df.copy()

    df = df.join(pd.get_dummies(df["accident_severity"], dtype=float))
    return df


def w_avg(row, weights):
    w1, w2 = weights

    values_with_w_sum = (
        row["injury_or_fatal_sum"] * w1 + row["material_damage_only_sum"] * w2
    )

    return values_with_w_sum / (w1 + w2)


def calc_weighted_mean_of_acc_severity(df):
    df = df.copy()
    df = transform_acc_sev_col_to_encoding(df)

    new_df = (
        df.groupby(["street"])
        .agg(
            injury_or_fatal_sum=("injury or fatal", "sum"),
            material_damage_only_sum=("material damage only", "sum"),
        )
        .reset_index()
    )

    new_df["weighted_avg"] = new_df.apply(lambda row: w_avg(row, [2, 1]), axis=1)

    print(new_df["weighted_avg"].describe())
    return new_df


# In[88]:


streets_with_accidents_ratio_df = calc_weighted_mean_of_acc_severity(accidents_17_23_df)


# In[89]:


streets_with_accidents_ratio_df.head(10)


# ### Let's merge safe_driving_df with strees with accidents ratio

# In[90]:


safe_driving_with_accidents_df = safe_driving_df.copy().merge(
    streets_with_accidents_ratio_df, how="left", left_on="road_name", right_on="street"
)


# ### This will allow to create Y variable labeling for our dataset

# In[91]:


plot_value_distributions_in_df(
    safe_driving_with_accidents_df[["weighted_avg"]], columns_to_avoid=[]
)


# In[92]:


safe_driving_with_accidents_df[["weighted_avg"]].describe()


# In[95]:


#### Preprocess y_var to binary values


# In[96]:


mean_weighted_avg = safe_driving_with_accidents_df["weighted_avg"].mean()


safe_driving_with_accidents_df["y_var"] = np.where(
    safe_driving_with_accidents_df["weighted_avg"] < mean_weighted_avg,
    "low-risk",
    "high-risk",
)


# ### Let's drop columns which are not important after the merge

# In[97]:


drop_columns_in_df(safe_driving_with_accidents_df, ["street"])


# In[98]:


safe_driving_with_accidents_df.head(10)


# ### This is the dataframe for modelling:
#

# In[99]:


safe_driving_with_accidents_df.head(10)


# In[100]:


def export_cleaned_data_to_csv(df_list=[], df_names=[]):
    for idx, df in enumerate(df_list):
        df.to_csv(f"./data_cleaned/{df_names[idx]}.csv", index=False)


# In[101]:


export_cleaned_data_to_csv(
    [safe_driving_with_accidents_df], ["safe_driving_with_accidents"]
)
