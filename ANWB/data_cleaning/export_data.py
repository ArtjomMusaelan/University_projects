#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import errors


# In[2]:


### Setup DB connection


# In[3]:


def connect_to_database():
    db_params = {
        "host": "194.171.191.226",
        "port": "6379",
        "database": "postgres",
        "user": "group6",
        "password": "blockd_2024group6_79",
    }
    try:
        conn_psycopg2 = psycopg2.connect(**db_params)
        print("Connection was successful!")
        return conn_psycopg2
    except Exception as e:
        print("Connection was not successful!")
        print(e)


# In[4]:


def create_cursor(connection):
    return connection.cursor()


def close_cursor(cursor):
    cursor.close()


def close_connection(connection):
    connection.close()


# In[5]:


def init_database_connection(func):
    def wrapper(*args, **kwargs):
        connection = connect_to_database()

        cursor = create_cursor(connection)

        res = func(cursor, *args, **kwargs)

        if kwargs.get("commit", False):
            connection.commit()

        close_cursor(cursor)
        close_connection(connection)
        return res

    return wrapper


# In[6]:


@init_database_connection
def make_query(cursor, query, show_results=False, commit=False):
    try:

        cursor.execute(query)

    except errors.DuplicateTable as e:
        print(e)
        print(
            "The table already exists but since this is a View creation it is allowed"
        )

    except Exception as e:
        print(e)
        return

    finally:

        if show_results:
            rows = cursor.fetchall()
            return rows

        return "Query succeeded"


# In[7]:


def get_column_names(table_name):
    q = f"""
    SELECT COLUMN_NAME
    FROM information_schema.columns
    WHERE table_schema ='group6_warehouse'
    AND table_name ='{table_name}'
    ORDER BY ordinal_position
    """
    return np.array(make_query(q, show_results=True)).flatten()


# ### Drop all the views if the script is rerun due to data changes
#

# In[8]:


def drop_views():
    make_query(
        """
DROP VIEW group6_warehouse.safe_driving 

""",
        show_results=False,
        commit=True,
    )

    make_query(
        """
DROP VIEW group6_warehouse.wind

""",
        show_results=False,
        commit=True,
    )

    make_query(
        """
DROP VIEW group6_warehouse.precipitation 

""",
        show_results=False,
        commit=True,
    )

    make_query(
        """
DROP VIEW group6_warehouse.temperature

""",
        show_results=False,
        commit=True,
    )

    make_query(
        """
    DROP VIEW group6_warehouse.accident_data_17_23 

    """,
        show_results=False,
        commit=True,
    )


# In[9]:


drop_views()

# ### Let's create views to store date on team's warehouse with columns we actually need
#

# In[10]:


make_query(
    """
CREATE VIEW group6_warehouse.safe_driving AS
SELECT *
FROM data_lake.safe_driving

""",
    show_results=False,
    commit=True,
)

# In[11]:


make_query(
    """
CREATE VIEW group6_warehouse.accident_data_17_23 AS
SELECT *
FROM data_lake.accident_data_17_23

""",
    show_results=False,
    commit=True,
)

# In[12]:


make_query(
    """
CREATE VIEW group6_warehouse.precipitation AS
SELECT DTG,RI_PWS_10
FROM data_lake.precipitation

""",
    show_results=False,
    commit=True,
)

# In[13]:


make_query(
    """
CREATE VIEW group6_warehouse.temperature AS
SELECT DTG,T_DRYB_10
FROM data_lake.temperature;

""",
    show_results=False,
    commit=True,
)

# In[14]:


make_query(
    """
CREATE VIEW group6_warehouse.wind AS
SELECT DTG,FF_SENSOR_10
FROM data_lake.wind;

""",
    show_results=False,
    commit=True,
)


# In[17]:


def load_sql_to_df(table_name):
    col_names = get_column_names(table_name)

    fetch_query = f"""
     SELECT * FROM group6_warehouse.{table_name}
     ;

    """

    result = make_query(fetch_query, show_results=True)

    df = pd.DataFrame(columns=col_names.tolist(), data=result)

    df.to_csv(f"./../../../data/original_data/{table_name}.csv", index=False)

    return df


# In[18]:


load_sql_to_df("safe_driving")

# In[17]:


load_sql_to_df("precipitation")

# In[18]:


load_sql_to_df("wind")

# In[19]:


load_sql_to_df("temperature")

# In[20]:


load_sql_to_df("accident_data_17_23")

# Data is exported to csv files
