import pandas as pd
import numpy as np


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

        Attributes
        ----------
        df : pandas.DataFrame
            Stores the loaded dataset.
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

            # Only convert if column
            # is not entirely NaN

            if not converted.isnull().all():

                self.df[col] = converted


        print("Automatic type conversion completed.")


    def data_summary(self):

        """
        Display important dataset information.

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


        print("\n========== DATA SUMMARY ==========")

        print(f"\nRows: {summary['Rows']}")

        print(f"Columns: {summary['Columns']}")

        print("\nNumeric Columns:")
        print(summary["Numeric Columns"])

        print("\nCategorical Columns:")
        print(summary["Categorical Columns"])

        print("\nMissing Values:")
        print(summary["Missing Values"])

        print("\nFirst 20 Rows:")
        print(self.df.head(20))

        return summary


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

                print(f"Could not process column {col}: {e}")


        print(f"Missing values handled using '{strategy}' strategy.")


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

        print(f"{removed} duplicate rows removed.")


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

            print("Selected column is not numeric.")
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


        print(f"Number of outliers: {len(outliers)}")


        if remove:

            self.df = self.df[
                (self.df[column] >= lower_bound) &
                (self.df[column] <= upper_bound)
            ]

            print("Outliers removed.")


        return outliers


    def get_dataframe(self):

        """
        Return current dataframe.

        Returns
        -------
        pandas.DataFrame
        """

        return self.df