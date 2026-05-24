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


        print("\n========== DATA SUMMARY ==========\n")

        print(
            f"Rows: {summary['Rows']} | "
            f"Columns: {summary['Columns']}"
        )

        print(
            f"\nNumerical Columns ({len(numeric_cols)}):"
        )

        print(numeric_cols)

        print(
            f"\nCategorical Columns "
            f"({len(categorical_cols)}):"
        )

        print(categorical_cols)


        missing_df = pd.DataFrame({

            "Column":
            self.df.columns,

            "Missing Values":
            self.df.isnull().sum().values
        })


        try:

            from IPython.display import display # pyright: ignore[reportMissingModuleSource]

            print("\nMissing Value Report:\n")

            display(
                missing_df[
                    missing_df["Missing Values"] > 0
                ]
            )

            print("\nDataset Preview:\n")

            display(
                self.df.head(preview_rows)
            )

        except Exception:

            print(missing_df)

            print(self.df.head(preview_rows))


        return summary


    def show_missing_data(self):

        """
        Display missing value report.
        """

        if self.df is None:

            print("No dataset loaded.")
            return


        missing = pd.DataFrame({

            "Column":
            self.df.columns,

            "Missing Values":
            self.df.isnull().sum().values,

            "Missing Percentage":
            (
                self.df.isnull().sum().values
                / len(self.df)
            ) * 100
        })


        missing = missing[
            missing["Missing Values"] > 0
        ]


        if missing.empty:

            print("No missing values found.")

            return


        try:

            from IPython.display import display # pyright: ignore[reportMissingModuleSource]

            display(
                missing
                .sort_values(
                    by="Missing Values",
                    ascending=False
                )
                .style
                .background_gradient()
            )

        except Exception:

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

            display(
                details.style.background_gradient()
            )

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

            print(f"\n========== {col} ==========\n")


            summary_df = pd.DataFrame({

                "Category":
                categorical[col]
                .value_counts(dropna=False)
                .index,

                "Count":
                categorical[col]
                .value_counts(dropna=False)
                .values
            })


            summary_df["Percentage"] = (

                summary_df["Count"]
                / len(categorical)

            ) * 100


            try:

                from IPython.display import display # pyright: ignore[reportMissingModuleSource]

                display(
                    summary_df
                    .style
                    .background_gradient()
                )

            except Exception:

                print(summary_df)


    def handle_missing_values(
            self,
            strategy='mean',
            columns=None,
            constant_value=None):

        """
        Handle missing values using:
        mean, median, mode, constant.
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
        Delete rows using indices.
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


        try:

            from IPython.display import display # pyright: ignore[reportMissingModuleSource]

            display(
                outliers.head(20)
            )

        except Exception:

            print(outliers.head(20))


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


        try:

            from IPython.display import display # pyright: ignore[reportMissingModuleSource]

            display(
                scaled_df.head(20)
                .style
                .background_gradient()
            )

        except Exception:

            print(scaled_df.head())


        return scaled_df


    def extract_normalized_categorical_data(
            self,
            method='onehot'):

        """
        Encode categorical columns.
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


        try:

            from IPython.display import display # pyright: ignore[reportMissingModuleSource]

            display(
                encoded_df.head(20)
                .style
                .background_gradient()
            )

        except Exception:

            print(encoded_df.head())


        return encoded_df


    def merge_processed_data(
            self,
            numeric_method='minmax',
            categorical_method='onehot'):

        """
        Merge normalized numeric data
        and encoded categorical data.
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


        try:

            from IPython.display import display # pyright: ignore[reportMissingModuleSource]

            display(
                merged_df.head(20)
                .style
                .background_gradient()
            )

        except Exception:

            print(merged_df.head())


        return merged_df


    def get_dataframe(self):

        """
        Return current dataframe.
        """

        return self.df