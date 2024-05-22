# mostly the code from Kaggle ML class
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# all functions
def score_dataset(X_train, X_valid, y_train, y_valid):
    # Function for comparing different approaches
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


def subsets_drop_col_missing(X_train, X_valid):
    # input: data of training & validation subsets
    # how to get subsets: use train_test_split from sklearn.model_selection
    # return: subsets with cols with missing values dropped
    cols_with_missing = [col for col in X_train.columns
                         if X_train[col].isnull().any()]
    reduced_X_train = X_train.drop(cols_with_missing, axis=1)
    reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)
    return reduced_X_train, reduced_X_valid


def subsets_imputation(X_train, X_valid):
    # input: data of training & validation subsets
    # how to get subsets: use train_test_split from sklearn.model_selection
    # return: subsets with cols with missing values imputed
    my_imputer = SimpleImputer()
    imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
    imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))
    imputed_X_train.columns = X_train.columns
    imputed_X_valid.columns = X_valid.columns
    return imputed_X_train, imputed_X_valid


def subsets_imputation_track(X_train, X_valid):
    # input: data of training & validation subsets
    # how to get subsets: use train_test_split from sklearn.model_selection
    # additional cols with be added _was_missing for identification purposes
    # return: subsets with cols with missing values imputed
    X_train_plus = X_train.copy()
    X_valid_plus = X_valid.copy()
    cols_with_missing = [col for col in X_train.columns
                         if X_train[col].isnull().any()]
    for col in cols_with_missing:
        X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
        X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()
    my_imputer = SimpleImputer()
    imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
    imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))
    imputed_X_train_plus.columns = X_train_plus.columns
    imputed_X_valid_plus.columns = X_valid_plus.columns
    return imputed_X_train_plus, imputed_X_valid_plus


def get_numerical_cols(X_train):
    numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]
    return numerical_cols


def get_categorical_cols(X_train):
    s = (X_train.dtypes == 'object')
    object_cols = list(s[s].index)
    return object_cols


def categorical_one_hot(X_train, X_valid):
    # intro: One-hot encoding creates new columns indicating the presence (or absence) of each possible value in the original data.
    # get object columns
    object_cols = get_categorical_cols(X_train)

    # Apply one-hot encoder to each column with categorical data
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

    # One-hot encoding removed index; put it back
    OH_cols_train.index = X_train.index
    OH_cols_valid.index = X_valid.index

    # Remove categorical columns (will replace with one-hot encoding)
    num_X_train = X_train.drop(object_cols, axis=1)
    num_X_valid = X_valid.drop(object_cols, axis=1)

    # Add one-hot encoded columns to numerical features
    OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

    # Ensure all columns have string type
    OH_X_train.columns = OH_X_train.columns.astype(str)
    OH_X_valid.columns = OH_X_valid.columns.astype(str)

    return OH_X_train, OH_X_valid


def categorical_ordinal(X_train, X_valid):
    # get object columns
    object_cols = get_categorical_cols(X_train)

    # Make copy to avoid changing original data
    label_X_train = X_train.copy()
    label_X_valid = X_valid.copy()

    # Apply ordinal encoder to each column with categorical data
    ordinal_encoder = OrdinalEncoder()
    label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
    label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])
    return label_X_train, label_X_valid


