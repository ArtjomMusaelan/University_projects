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
from pathlib import Path


# In[2]:
def show_eda_analysis():

    def import_df(name="", columns_to_convert=[], custom_path=False):

        path = f"{Path(__file__).resolve().parent.parent.parent.parent}/data/original_data/{name}.csv"
        if custom_path:
            path = custom_path
        df = pd.read_csv(path)

        for col in columns_to_convert:
            df[col] = pd.to_datetime(df[col])
        return df

    # In[3]:

    safe_driving_df = import_df(
        "safe_driving_with_accidents", ["event_start", "event_end"]
    )

    ### Let's limit datafram to avoid slowing down the plotting and still capture most important data features

    safe_driving_df = safe_driving_df.iloc[:10000, :]

    # In[4]:

    accidents_17_23_df = import_df(
        custom_path=f"{Path(__file__).resolve().parent.parent.parent.parent}/data/original_data/accident_data_17_23.csv"
    )

    # In[5]:

    accidents_17_23_df.dtypes

    # # Exploratory Data Analysis (EDA)

    # In[6]:

    # Show general information about the dataframe
    def show_dataframe_general_info(df):
        print("General info of df")
        print(df.info())
        print("Description of df")
        print(df.describe())

    show_dataframe_general_info(safe_driving_df)

    # In[7]:

    # Check for missing values
    def check_df_missing_values(df):
        total_missing_values = df.isna().sum().sum()
        print(f"Total number of missing values: {total_missing_values}")
        if total_missing_values > 0:
            print("Number of missing values in particular columns:")
            print(df.isna().sum())

    check_df_missing_values(safe_driving_df)

    # In[8]:

    # Define necessary numerical features
    important_numerical_features = [
        "duration_seconds",
        "latitude",
        "longitude",
        "speed_kmh",
        "end_speed_kmh",
        "maxwaarde",
        "last_hour_wind_avg",
        "last_hour_temp_avg",
        "last_hour_rain_avg",
    ]

    # In[9]:

    # Temporal analysis
    safe_driving_df["event_start"] = pd.to_datetime(safe_driving_df["event_start"])
    safe_driving_df["event_end"] = pd.to_datetime(safe_driving_df["event_end"])

    plt.figure(figsize=(12, 6))
    safe_driving_df["hour"] = safe_driving_df["event_start"].dt.hour
    sns.countplot(x="hour", data=safe_driving_df)
    plt.title("Incidents by Hour of Day")
    plt.show()

    # In[10]:

    # Plot value distributions
    def plot_value_distributions(df):
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        for col in important_numerical_features:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.subplot(1, 2, 2)
            sns.boxplot(x=df[col])
            plt.title(f"Box plot of {col}")
            plt.show()

    plot_value_distributions(safe_driving_df)

    # In[11]:

    # Correlation heatmap
    plt.figure(figsize=(16, 10))
    correlation_matrix = safe_driving_df[important_numerical_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

    # In[12]:

    # Pair plots for important numerical features
    sns.pairplot(safe_driving_df[important_numerical_features])
    plt.show()

    # In[13]:

    # Distribution of all categorical features

    categorical_cols = safe_driving_df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        plt.figure(figsize=(12, 6))
        if col == "road_name":
            top_20_roads = (
                safe_driving_df["road_name"].value_counts().nlargest(20).index
            )
            sns.countplot(
                x=safe_driving_df[safe_driving_df["road_name"].isin(top_20_roads)][
                    "road_name"
                ]
            )
            plt.title("Distribution of Top 20 Road Names")
        else:
            sns.countplot(x=safe_driving_df[col])
            plt.title(f"Distribution of {col}")
        plt.xticks(rotation=90)
        plt.show()

    # In[ ]:

    # Box plots for numerical features across all categorical features
    categorical_cols = safe_driving_df.select_dtypes(include=["object"]).columns

    for cat_col in categorical_cols:
        for num_col in important_numerical_features:
            if num_col in safe_driving_df.columns:
                plt.figure(figsize=(12, 6))
                if cat_col == "road_name":
                    top_20_roads = (
                        safe_driving_df["road_name"].value_counts().nlargest(20).index
                    )
                    sns.boxplot(
                        x=safe_driving_df[
                            safe_driving_df["road_name"].isin(top_20_roads)
                        ][cat_col],
                        y=safe_driving_df[num_col],
                    )
                    plt.title(f"{num_col} distribution across Top 20 {cat_col}")
                else:
                    sns.boxplot(x=safe_driving_df[cat_col], y=safe_driving_df[num_col])
                    plt.title(f"{num_col} distribution across {cat_col}")
                plt.xticks(rotation=90)
                plt.show()

    # ### EDA for aditional dataset

    # In[15]:

    accidents_17_23_df.head(10)

    # In[16]:

    show_dataframe_general_info(accidents_17_23_df)

    # In[17]:

    # Plot value distributions
    def plot_value_distributions_2(df):
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        for col in numeric_cols:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.show()

    plot_value_distributions_2(accidents_17_23_df)

    # In[18]:

    # Plot value distributions
    def plot_categorical_value_distributions(df):
        categorical_columns = df.select_dtypes(include=["object"]).columns
        for col in categorical_columns:
            plt.figure(figsize=(12, 6))
            if col == "street":
                top_20 = df[col].value_counts().nlargest(20)
                sns.barplot(x=top_20.index, y=top_20.values)
                plt.tick_params(axis="x", rotation=45)
                plt.title(f"Top 20 most frequent accidents in {col}")
            else:
                plt.subplot(1, 2, 1)
                sns.histplot(df[col], kde=True)
                plt.tick_params(axis="x", rotation=45)
                plt.title(f"Distribution of {col}")
                plt.show()

    plot_categorical_value_distributions(accidents_17_23_df)

    # In[19]:

    from scipy.stats import chi2_contingency

    def cramers_v(x, y):
        """Calculate Cramér's V statistic for categorical-categorical association."""
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        r, k = confusion_matrix.shape
        return np.sqrt(chi2 / (n * (min(r, k) - 1)))

    def calculate_cramers_v_matrix(df, cols):
        """Calculate Cramér's V matrix for a given DataFrame and columns."""
        cramers_v_matrix = pd.DataFrame(index=cols, columns=cols)
        for col1 in cols:
            for col2 in cols:
                if col1 == col2:
                    cramers_v_matrix.loc[col1, col2] = 1.0
                else:
                    cramers_v_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])
        cramers_v_matrix = cramers_v_matrix.astype(float)
        return cramers_v_matrix

    # Select categorical columns
    categorical_columns = accidents_17_23_df.select_dtypes(include=["object"]).columns

    # Calculate Cramér's V matrix
    cramers_v_matrix = calculate_cramers_v_matrix(
        accidents_17_23_df, categorical_columns
    )

    # Plot the heatmap
    plt.figure(figsize=(16, 10))
    sns.heatmap(cramers_v_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Cramér's V Correlation Heatmap")
    plt.show()

    # In[ ]:

    # In[ ]:
