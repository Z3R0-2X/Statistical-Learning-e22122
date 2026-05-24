import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    LabelEncoder
)


class DataInspector:

    """
    A reusable toolkit for:
    - data loading
    - data cleaning
    - preprocessing
    - dataset inspection
    """

    def __init__(self):

        """
        Constructor method.
        """

        self.df = None


    def upload_data(self, source):

        """
        Load dataset from:
        - local CSV file
        - online CSV URL

        Parameters
        ----------
        source : str
            File path or URL

        Returns
        -------
        pandas.DataFrame
        """

        try:

            self.df = pd.read_csv(source)

            print("Dataset loaded successfully.")

            return self.df

        except Exception as e:

            print(f"Error loading dataset: {e}")


    def clean_garbage_values(self):

        """
        Replace garbage strings with NaN values.
        """

        if self.df is None:

            print("No dataset loaded.")
            return


        garbage_values = [
            '?',
            '??',
            'n/a',
            'N/A',
            'NULL',
            'null',
            'NaN',
            'nan',
            '',
            ' '
        ]


        self.df.replace(
            garbage_values,
            np.nan,
            inplace=True
        )

        print("Garbage values cleaned.")


    def auto_convert_types(self):

        """
        Automatically convert numeric-like
        columns into numeric datatypes.
        """

        if self.df is None:

            print("No dataset loaded.")
            return


        for col in self.df.columns:

            converted = pd.to_numeric(
                self.df[col],
                errors='coerce'
            )


            if not converted.isnull().all():

                self.df[col] = converted


        print("Automatic type conversion completed.")


    def data_summary(self, preview_rows=20):

        """
        Display important dataset information.

        Parameters
        ----------
        preview_rows : int
            Number of preview rows

        Returns
        -------
        dict
        """

        if self.df is None:

            print("No dataset loaded.")
            return


        numeric_cols = self.df.select_dtypes(
            include=np.number
        ).columns.tolist()


        categorical_cols = self.df.select_dtypes(
            exclude=np.number
        ).columns.tolist()


        summary = {

            "Rows":
            self.df.shape[0],

            "Columns":
            self.df.shape[1],

            "Numeric Columns":
            numeric_cols,

            "Categorical Columns":
            categorical_cols,

            "Missing Values":
            self.df.isnull().sum().to_dict()
        }


        print("\n--- Data Summary ---\n")

        print(
            f"Rows: {summary['Rows']} | "
            f"Columns: {summary['Columns']}"
        )

        print(
            f"Numerical ({len(numeric_cols)}): "
            f"{numeric_cols}"
        )

        print(
            f"Categorical ({len(categorical_cols)}): "
            f"{categorical_cols}"
        )

        print("\nMissing Values:")
        print(summary["Missing Values"])


        try:

            from IPython.display import display # pyright: ignore[reportMissingModuleSource]

            display(
                self.df.head(preview_rows)
            )

        except Exception:

            print(
                self.df.head(preview_rows)
            )


        return summary


    def show_missing_data(self):

        """
        Display missing values in each column.
        """

        if self.df is None:

            print("No dataset loaded.")
            return


        missing = self.df.isnull().sum()

        missing = missing[missing > 0]


        if missing.empty:

            print("No missing values found.")

        else:

            print(missing)


    def column_details(self):

        """
        Display detailed information
        about all columns.
        """

        if self.df is None:

            print("No dataset loaded.")
            return


        details = pd.DataFrame({

            "Column":
            self.df.columns,

            "Datatype":
            self.df.dtypes.values,

            "Missing Values":
            self.df.isnull().sum().values,

            "Unique Values":
            self.df.nunique().values
        })


        try:

            from IPython.display import display # pyright: ignore[reportMissingModuleSource]

            display(details)

        except Exception:

            print(details)


    def get_categorical_summary(self):

        """
        Display summary of categorical columns.
        """

        if self.df is None:

            print("No dataset loaded.")
            return


        categorical = self.df.select_dtypes(
            exclude=np.number
        )


        if categorical.empty:

            print("No categorical columns found.")
            return


        for col in categorical.columns:

            print(f"\n--- {col} ---")

            print(
                categorical[col]
                .value_counts(dropna=False)
            )


    def handle_missing_values(
            self,
            strategy='mean',
            columns=None,
            constant_value=None):

        """
        Handle missing values using different strategies.

        Parameters
        ----------
        strategy : str
            mean, median, mode, constant

        columns : list
            Columns to process

        constant_value :
            Value used when strategy='constant'
        """

        if self.df is None:

            print("No dataset loaded.")
            return


        if columns is None:

            columns = self.df.columns


        for col in columns:

            try:

                if strategy == 'mean':

                    self.df[col].fillna(
                        self.df[col].mean(),
                        inplace=True
                    )


                elif strategy == 'median':

                    self.df[col].fillna(
                        self.df[col].median(),
                        inplace=True
                    )


                elif strategy == 'mode':

                    self.df[col].fillna(
                        self.df[col].mode()[0],
                        inplace=True
                    )


                elif strategy == 'constant':

                    self.df[col].fillna(
                        constant_value,
                        inplace=True
                    )

            except Exception as e:

                print(
                    f"Could not process column {col}: {e}"
                )


        print(
            f"Missing values handled using "
            f"'{strategy}' strategy."
        )


    def remove_duplicates(self):

        """
        Remove duplicate rows.
        """

        if self.df is None:

            print("No dataset loaded.")
            return


        before = len(self.df)

        self.df.drop_duplicates(
            inplace=True
        )

        after = len(self.df)

        removed = before - after

        print(
            f"{removed} duplicate rows removed."
        )


    def delete_columns(self, columns):

        """
        Delete columns from dataset.

        Parameters
        ----------
        columns : list
            List of column names
        """

        if self.df is None:

            print("No dataset loaded.")
            return


        self.df.drop(
            columns=columns,
            inplace=True,
            errors='ignore'
        )

        print("Columns deleted successfully.")


    def delete_rows(self, row_indices):

        """
        Delete rows using index values.

        Parameters
        ----------
        row_indices : list
            List of row indices
        """

        if self.df is None:

            print("No dataset loaded.")
            return


        self.df.drop(
            index=row_indices,
            inplace=True,
            errors='ignore'
        )

        print("Rows deleted successfully.")


    def handle_outliers(
            self,
            column,
            remove=True):

        """
        Detect and optionally remove outliers
        using IQR method.

        Parameters
        ----------
        column : str
            Numeric column name

        remove : bool
            If True, remove outliers

        Returns
        -------
        pandas.DataFrame
            Outlier rows
        """

        if self.df is None:

            print("No dataset loaded.")
            return


        if column not in self.df.columns:

            print("Column not found.")
            return


        if not pd.api.types.is_numeric_dtype(
                self.df[column]):

            print(
                "Selected column is not numeric."
            )

            return


        Q1 = self.df[column].quantile(0.25)

        Q3 = self.df[column].quantile(0.75)

        IQR = Q3 - Q1


        lower_bound = Q1 - 1.5 * IQR

        upper_bound = Q3 + 1.5 * IQR


        outliers = self.df[
            (self.df[column] < lower_bound) |
            (self.df[column] > upper_bound)
        ]


        print(
            f"Number of outliers: {len(outliers)}"
        )


        if remove:

            self.df = self.df[
                (self.df[column] >= lower_bound) &
                (self.df[column] <= upper_bound)
            ]

            print("Outliers removed.")


        return outliers


    def extract_normalized_numeric_data(
            self,
            method='minmax'):

        """
        Normalize numeric columns.

        Parameters
        ----------
        method : str
            minmax, standard, robust

        Returns
        -------
        pandas.DataFrame
        """

        if self.df is None:

            print("No dataset loaded.")
            return


        numeric_df = self.df.select_dtypes(
            include=np.number
        )


        if numeric_df.empty:

            print("No numeric columns found.")
            return


        if method == 'minmax':

            scaler = MinMaxScaler()


        elif method == 'standard':

            scaler = StandardScaler()


        elif method == 'robust':

            scaler = RobustScaler()


        else:

            print("Invalid normalization method.")
            return


        scaled_data = scaler.fit_transform(
            numeric_df
        )


        scaled_df = pd.DataFrame(
            scaled_data,
            columns=numeric_df.columns
        )


        print(
            f"{method} normalization completed."
        )

        return scaled_df


    def extract_normalized_categorical_data(
            self,
            method='onehot'):

        """
        Encode categorical columns.

        Parameters
        ----------
        method : str
            onehot or ordinal

        Returns
        -------
        pandas.DataFrame
        """

        if self.df is None:

            print("No dataset loaded.")
            return


        categorical_df = self.df.select_dtypes(
            exclude=np.number
        )


        if categorical_df.empty:

            print("No categorical columns found.")
            return


        if method == 'onehot':

            encoded_df = pd.get_dummies(
                categorical_df,
                drop_first=True
            )


        elif method == 'ordinal':

            encoded_df = categorical_df.copy()


            encoder = LabelEncoder()


            for col in encoded_df.columns:

                encoded_df[col] = encoder.fit_transform(
                    encoded_df[col].astype(str)
                )


        else:

            print("Invalid encoding method.")
            return


        print(
            f"{method} encoding completed."
        )

        return encoded_df


    def merge_processed_data(
            self,
            numeric_method='minmax',
            categorical_method='onehot'):

        """
        Merge normalized numeric data
        and encoded categorical data.

        Returns
        -------
        pandas.DataFrame
        """

        numeric_df = (
            self.extract_normalized_numeric_data(
                method=numeric_method
            )
        )


        categorical_df = (
            self.extract_normalized_categorical_data(
                method=categorical_method
            )
        )


        merged_df = pd.concat(
            [numeric_df, categorical_df],
            axis=1
        )


        print(
            "Processed dataset merged successfully."
        )

        return merged_df


    def get_dataframe(self):

        """
        Return current dataframe.

        Returns
        -------
        pandas.DataFrame
        """

        return self.df