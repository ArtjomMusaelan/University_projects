import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def import_df(name="", columns_to_convert=[], custom_path=False):
    path = f"./../data_cleaning/data_cleaned/{name}.csv"
    if custom_path:
        path = custom_path
    df = pd.read_csv(path)

    for col in columns_to_convert:
        df[col] = pd.to_datetime(df[col])
    return df


safe_driving_with_accidents_df = import_df(
    "safe_driving_with_accidents", ["event_start", "event_end"]
).iloc[:500, :]

### 1 try no limits (90 000 rows) if longer than 20 minutes switch to second try

### 2 try with 50 000
### 3 try with 20 000


### RandomForestClassifier


X_droplist = [
    "y_var",
    "event_start",
    "event_end",
    "eventid",
    "incident_severity",
    "weighted_avg",
    "material_damage_only_sum",
    "road_segment_id",
    "injury_or_fatal_sum",
]
X = safe_driving_with_accidents_df.drop(columns=X_droplist)
y = safe_driving_with_accidents_df["y_var"]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

# Create a column transformer with imputation and encoding
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numerical_cols,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_cols,
        ),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a pipeline with preprocessing and model
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100)),
    ]
)

# Train the model
model_pipeline.fit(X_train, y_train)

# Evaluate the model using cross-validation
cv_scores = cross_val_score(model_pipeline, X, y, cv=5, scoring="accuracy")
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean()}")

# Predict on the test set
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
### XGBClassifier


X_droplist = [
    "y_var",
    "event_start",
    "event_end",
    "eventid",
    "incident_severity",
    "weighted_avg",
    "material_damage_only_sum",
    "road_segment_id",
    "injury_or_fatal_sum",
    "road_name",
]
X = safe_driving_with_accidents_df.drop(columns=X_droplist)
y = safe_driving_with_accidents_df["y_var"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

# Create a column transformer with imputation and encoding
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numerical_cols,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_cols,
        ),
    ]
)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Create a pipeline with preprocessing and model
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            XGBClassifier(
                n_estimators=100,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
            ),
        ),
    ]
)

# Train the model
model_pipeline.fit(X_train, y_train)

# Evaluate the model using cross-validation
cv_scores = cross_val_score(model_pipeline, X, y_encoded, cv=5, scoring="accuracy")
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean()}")
from sklearn.metrics import accuracy_score, classification_report

# Predict on the test set
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [3, 6, 9],
    "classifier__learning_rate": [0.01, 0.1, 0.2],
    "classifier__subsample": [0.6, 0.8, 1.0],
}

# Perform grid search
grid_search = GridSearchCV(
    estimator=model_pipeline, param_grid=param_grid, cv=5, scoring="accuracy"
)
grid_search.fit(X_train, y_train)

# Get the best model and best parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Print the best parameters
print("Best Parameters:", best_params)

# Evaluate the best model
y_pred_best = best_model.predict(X_test)
print("Best Model Accuracy:", accuracy_score(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best, target_names=label_encoder.classes_))
### XGBClassifier, with extracted data from event_start

# Extract year, month, day, and hour from 'event_start'
safe_driving_with_accidents_df["event_start"] = pd.to_datetime(
    safe_driving_with_accidents_df["event_start"]
)
safe_driving_with_accidents_df["year"] = safe_driving_with_accidents_df[
    "event_start"
].dt.year
safe_driving_with_accidents_df["month"] = safe_driving_with_accidents_df[
    "event_start"
].dt.month
safe_driving_with_accidents_df["day"] = safe_driving_with_accidents_df[
    "event_start"
].dt.day
safe_driving_with_accidents_df["hour"] = safe_driving_with_accidents_df[
    "event_start"
].dt.hour

# Drop columns that are not used
X_droplist = [
    "y_var",
    "event_start",
    "event_end",
    "eventid",
    "incident_severity",
    "weighted_avg",
    "material_damage_only_sum",
    "road_segment_id",
    "injury_or_fatal_sum",
    "road_name",
    "latitude",
    "longitude",
]
X = safe_driving_with_accidents_df.drop(columns=X_droplist)
y = safe_driving_with_accidents_df["y_var"]

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

# Create a column transformer with imputation and encoding
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numerical_cols,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_cols,
        ),
    ]
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Create a pipeline with preprocessing and model
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        ),
    ]
)

# Train the model
model_pipeline.fit(X_train, y_train)

# Evaluate the model using cross-validation
cv_scores = cross_val_score(model_pipeline, X, y_encoded, cv=5, scoring="accuracy")
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean()}")

# Predict on the test set
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
# Define the parameter grid for GridSearchCV
param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [3, 6, 9],
    "classifier__learning_rate": [0.01, 0.1, 0.2],
    "classifier__subsample": [0.6, 0.8, 1.0],
}

# Create GridSearchCV object
grid_search = GridSearchCV(
    estimator=model_pipeline, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best model and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Print best parameters and best score
print("Best parameters:", best_params)
print("Best cross-validation score:", grid_search.best_score_)

# Predict on the test set with the best model
y_pred_best = best_model.predict(X_test)

# Evaluate the best model
print("Best Model Accuracy:", accuracy_score(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best, target_names=label_encoder.classes_))

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt="d")
plt.show()


### XGBClassifier, Dominic's code used, with coordinates, with road_name


# Transformer for target variable encoding
class TargetVariableTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_variable, is_regression):
        self.target_variable = target_variable
        self.is_regression = is_regression

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.is_regression:
            return X
        else:
            X[self.target_variable] = (
                X[self.target_variable].astype("category").cat.codes
            )
            return X


# Pipeline to transform the target variable
def target_variable_pipeline(target_variable, is_regression):
    gen_ppl = Pipeline(
        steps=[
            (
                "transform_target",
                TargetVariableTransformer(
                    target_variable=target_variable, is_regression=is_regression
                ),
            )
        ]
    )
    return gen_ppl


# Function to provide the full preprocessing and model pipeline
def provide_full_pipeline(model_obj, num_features, cat_features, target_variable, df):
    # Create column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="mean")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat_features,
            ),
        ]
    )

    # Combine preprocessing and model into a single pipeline
    model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model_obj)])

    return (
        model,
        df,
        {
            "num_features": num_features,
            "cat_features": cat_features,
            "target_variable": target_variable,
        },
    )


# Decorator function for model training and evaluation
def model_training(func):
    def wrapper(
        df,
        num_features,
        cat_features,
        target_variable,
        model_obj,
        is_regression=True,
        *args,
    ):
        # Create full pipeline and preprocess the data
        model, new_df, model_properties = provide_full_pipeline(
            model_obj,
            num_features=num_features,
            cat_features=cat_features,
            target_variable=target_variable,
            df=df,
        )

        new_df = target_variable_pipeline(target_variable, is_regression).fit_transform(
            new_df
        )

        # Split the data into features (X) and target (y)
        X = new_df.drop(target_variable, axis=1)
        y = new_df[target_variable]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train the model
        model.fit(X_train, y_train)

        # Call the evaluation function
        res = func(model, X_test, y_test, model_properties, *args)

        # Print evaluation metrics
        if is_regression:
            r2_score = model.score(X_test, y_test)
            rmse = mean_squared_error(y_test, model.predict(X_test)) ** (1 / 2)
            mae = mean_absolute_error(y_test, model.predict(X_test))

            print(f"R2 score: {r2_score}")
            print("MSE: ", rmse**2)
            print(f"RMSE: {rmse}")
            print(f"MAE: {mae}")
        else:
            score = model.score(X_test, y_test)
            print("Score", score)

            y_pred = model.predict(X_test)
            print(classification_report(y_test, y_pred))

            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d")
            plt.show()

        return res

    return wrapper


# Assuming safe_driving_with_accidents_df is your dataframe
df = safe_driving_with_accidents_df.copy()

# Define numerical and categorical features
num_features = df.select_dtypes(include=[np.number]).columns.tolist()
cat_features = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Remove target and other columns from features
X_droplist = [
    "y_var",
    "event_start",
    "event_end",
    "eventid",
    "incident_severity",
    "weighted_avg",
    "material_damage_only_sum",
    "road_segment_id",
    "injury_or_fatal_sum",
]
num_features = [feature for feature in num_features if feature not in X_droplist]
cat_features = [feature for feature in cat_features if feature not in X_droplist]

# Define target variable
target_variable = "y_var"

# Define model
model_obj = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")


# Evaluation function decorated with model_training
@model_training
def evaluate_model(model, X_test, y_test, model_properties, *args):
    return model.score(X_test, y_test)


# Manually set hyperparameters (similar to a simplified grid search)
model_obj.set_params(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8)

# Run model training and evaluation
result = evaluate_model(
    df, num_features, cat_features, target_variable, model_obj, is_regression=False
)
### XGBClassifier, without coordinates, without road_name

X_droplist = [
    "y_var",
    "event_start",
    "event_end",
    "eventid",
    "incident_severity",
    "weighted_avg",
    "material_damage_only_sum",
    "road_segment_id",
    "injury_or_fatal_sum",
    "road_name",
    "latitude",
    "longitude",
]
X = safe_driving_with_accidents_df.drop(columns=X_droplist)
y = safe_driving_with_accidents_df["y_var"]
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

# Create a column transformer with imputation and encoding
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numerical_cols,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_cols,
        ),
    ]
)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

# Create a pipeline with preprocessing and model
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            XGBClassifier(
                n_estimators=100,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
            ),
        ),
    ]
)

# Train the model
model_pipeline.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score

# Evaluate the model using cross-validation
cv_scores = cross_val_score(model_pipeline, X, y_encoded, cv=5, scoring="accuracy")
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean()}")
from sklearn.metrics import accuracy_score, classification_report

# Predict on the test set
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [3, 6, 9],
    "classifier__learning_rate": [0.01, 0.1, 0.2],
    "classifier__subsample": [0.6, 0.8, 1.0],
}

# Perform grid search
grid_search = GridSearchCV(
    estimator=model_pipeline, param_grid=param_grid, cv=5, scoring="accuracy"
)
grid_search.fit(X_train, y_train)

# Get the best model and best parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Print the best parameters
print("Best Parameters:", best_params)

# Evaluate the best model
y_pred_best = best_model.predict(X_test)
print("Best Model Accuracy:", accuracy_score(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best, target_names=label_encoder.classes_))
### XGBClassifier, Dominic's code used, without coordinates and road_name

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb


# Transformer for target variable encoding
class TargetVariableTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_variable, is_regression):
        self.target_variable = target_variable
        self.is_regression = is_regression

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.is_regression:
            return X
        else:
            X[self.target_variable] = (
                X[self.target_variable].astype("category").cat.codes
            )
            return X


# Pipeline to transform the target variable
def target_variable_pipeline(target_variable, is_regression):
    gen_ppl = Pipeline(
        steps=[
            (
                "transform_target",
                TargetVariableTransformer(
                    target_variable=target_variable, is_regression=is_regression
                ),
            )
        ]
    )
    return gen_ppl


# Function to provide the full preprocessing and model pipeline
def provide_full_pipeline(model_obj, num_features, cat_features, target_variable, df):
    # Create column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="mean")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat_features,
            ),
        ]
    )

    # Combine preprocessing and model into a single pipeline
    model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model_obj)])

    return (
        model,
        df,
        {
            "num_features": num_features,
            "cat_features": cat_features,
            "target_variable": target_variable,
        },
    )


# Decorator function for model training and evaluation
def model_training(func):
    def wrapper(
        df,
        num_features,
        cat_features,
        target_variable,
        model_obj,
        is_regression=True,
        *args,
    ):
        # Create full pipeline and preprocess the data
        model, new_df, model_properties = provide_full_pipeline(
            model_obj,
            num_features=num_features,
            cat_features=cat_features,
            target_variable=target_variable,
            df=df,
        )

        new_df = target_variable_pipeline(target_variable, is_regression).fit_transform(
            new_df
        )

        # Split the data into features (X) and target (y)
        X = new_df.drop(target_variable, axis=1)
        y = new_df[target_variable]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train the model
        model.fit(X_train, y_train)

        # Call the evaluation function
        res = func(model, X_test, y_test, model_properties, *args)

        # Print evaluation metrics
        if is_regression:
            r2_score = model.score(X_test, y_test)
            rmse = mean_squared_error(y_test, model.predict(X_test)) ** (1 / 2)
            mae = mean_absolute_error(y_test, model.predict(X_test))

            print(f"R2 score: {r2_score}")
            print("MSE: ", rmse**2)
            print(f"RMSE: {rmse}")
            print(f"MAE: {mae}")
        else:
            score = model.score(X_test, y_test)
            print("Score", score)

            y_pred = model.predict(X_test)
            print(classification_report(y_test, y_pred))

            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d")
            plt.show()

        return res

    return wrapper


import pandas as pd

# Assuming safe_driving_with_accidents_df is your dataframe
df = safe_driving_with_accidents_df.copy()

# Define numerical and categorical features
num_features = df.select_dtypes(include=[np.number]).columns.tolist()
cat_features = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Remove target and other columns from features
X_droplist = [
    "y_var",
    "event_start",
    "event_end",
    "eventid",
    "incident_severity",
    "weighted_avg",
    "material_damage_only_sum",
    "road_segment_id",
    "injury_or_fatal_sum",
    "road_name",
    "latitude",
    "longitude",
]
num_features = [feature for feature in num_features if feature not in X_droplist]
cat_features = [feature for feature in cat_features if feature not in X_droplist]

# Define target variable
target_variable = "y_var"

# Define model
model_obj = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")


# Evaluation function decorated with model_training
@model_training
def evaluate_model(model, X_test, y_test, model_properties, *args):
    return model.score(X_test, y_test)


# Manually set hyperparameters (similar to a simplified grid search)
model_obj.set_params(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8)

# Run model training and evaluation
result = evaluate_model(
    df, num_features, cat_features, target_variable, model_obj, is_regression=False
)
### RandomForestClassifier, without coordinates and road_name, with extracted data from event_start

# Extract year, month, day, and hour from 'event_start'
safe_driving_with_accidents_df["event_start"] = pd.to_datetime(
    safe_driving_with_accidents_df["event_start"]
)
safe_driving_with_accidents_df["year"] = safe_driving_with_accidents_df[
    "event_start"
].dt.year
safe_driving_with_accidents_df["month"] = safe_driving_with_accidents_df[
    "event_start"
].dt.month
safe_driving_with_accidents_df["day"] = safe_driving_with_accidents_df[
    "event_start"
].dt.day
safe_driving_with_accidents_df["hour"] = safe_driving_with_accidents_df[
    "event_start"
].dt.hour
X_droplist = [
    "y_var",
    "event_start",
    "event_end",
    "eventid",
    "incident_severity",
    "weighted_avg",
    "material_damage_only_sum",
    "road_segment_id",
    "injury_or_fatal_sum",
    "road_name",
    "latitude",
    "longitude",
]
X = safe_driving_with_accidents_df.drop(columns=X_droplist)
y = safe_driving_with_accidents_df["y_var"]
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

# Create a column transformer with imputation and encoding
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numerical_cols,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_cols,
        ),
    ]
)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Create a pipeline with preprocessing and model
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100)),
    ]
)

# Train the model
model_pipeline.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score

# Evaluate the model using cross-validation
cv_scores = cross_val_score(model_pipeline, X, y, cv=5, scoring="accuracy")
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean()}")
from sklearn.metrics import accuracy_score, classification_report

# Predict on the test set
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Define the parameter grid for GridSearchCV
param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [None, 10, 20, 30],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__bootstrap": [True, False],
}

# Create GridSearchCV object
grid_search = GridSearchCV(
    estimator=model_pipeline, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best model and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Print best parameters and best score
print("Best parameters:", best_params)
print("Best cross-validation score:", grid_search.best_score_)

# Predict on the test set with the best model
y_pred_best = best_model.predict(X_test)

# Evaluate the best model
print("Best Model Accuracy:", accuracy_score(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))
