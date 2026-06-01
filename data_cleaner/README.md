# DataInspector

A reusable Python toolkit for:

* Data Loading
* Data Cleaning
* Missing Value Handling
* Outlier Detection
* Feature Engineering
* Data Normalization
* Categorical Encoding
* Exploratory Data Analysis (EDA)
* Statistical Correlation Analysis
* Interactive Plotly Visualizations

Designed for use in:

* Google Colab
* Jupyter Notebook
* VS Code

---

## Installation

Install directly from GitHub:

```bash
!pip install git+https://github.com/Z3R0-2X/Statistical-Learning-e22122.git#subdirectory=data_cleaner
```

Update to the latest version:

```bash
!pip install --upgrade --force-reinstall git+https://github.com/Z3R0-2X/Statistical-Learning-e22122.git#subdirectory=data_cleaner
```

---

## Usage

```python
from datainspector import DataInspector

inspector = DataInspector()

```
# Custom Plotting Methods

The package includes a separate PlottingMethods class for creating
individual charts that can be used independently of DataInspector.

## Import

```python
from datainspector import PlottingMethods

plotter = PlottingMethods()
```

# Loading Data

Load a local CSV file:

```python
inspector.upload_data("data.csv")
```

Load a dataset from a URL:

```python
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

inspector.upload_data(url)
```

---

# Data Inspection

## Dataset Summary

Displays:

* Number of rows
* Number of columns
* Numerical columns
* Categorical columns
* Missing value report
* Dataset preview

```python
inspector.data_summary()
```

---

## Missing Value Analysis

```python
inspector.show_missing_data()
```

---

## Column Information

```python
inspector.column_details()
```

---

## Categorical Summary

```python
inspector.get_categorical_summary()
```

---

# Data Cleaning

## Clean Garbage Values

Automatically converts:

* ?
* ??
* n/a
* N/A
* NULL
* null
* NaN
* nan
* Empty strings

into proper NaN values.

```python
inspector.clean_garbage_values()
```

---

## Automatic Type Conversion

Automatically converts numeric-looking columns into numeric datatypes.

```python
inspector.auto_convert_types()
```

---

## Handle Missing Values

### Mean Imputation

```python
inspector.handle_missing_values(
    strategy="mean",
    columns=["Age"]
)
```

### Median Imputation

```python
inspector.handle_missing_values(
    strategy="median",
    columns=["Age"]
)
```

### Mode Imputation

```python
inspector.handle_missing_values(
    strategy="mode",
    columns=["Embarked"]
)
```

### Constant Value Imputation

```python
inspector.handle_missing_values(
    strategy="constant",
    columns=["Cabin"],
    constant_value="Unknown"
)
```

---

## Remove Duplicate Rows

```python
inspector.remove_duplicates()
```

---

## Delete Columns

```python
inspector.delete_columns([
    "PassengerId",
    "Ticket"
])
```

---

## Delete Rows

```python
inspector.delete_rows([
    0,
    1,
    2
])
```

---

## Outlier Detection

Detect Outliers Only

```python
inspector.handle_outliers(
    column="Fare",
    remove=False
)
```

Detect and Remove Outliers

```python
inspector.handle_outliers(
    column="Fare",
    remove=True
)
```

---

# Feature Engineering & Normalization

## Extract Numeric Data

```python
numeric_df = inspector.extract_numeric_data()
```

---

## Extract Categorical Data

```python
categorical_df = inspector.extract_categorical_data()
```

---

## Normalize Numeric Data

### Min-Max Scaling

```python
scaled_df = inspector.extract_normalized_numeric_data(
    method="minmax"
)
```

### Standard Scaling

```python
scaled_df = inspector.extract_normalized_numeric_data(
    method="standard"
)
```

### Robust Scaling

```python
scaled_df = inspector.extract_normalized_numeric_data(
    method="robust"
)
```

---

## Encode Categorical Data

### One-Hot Encoding

```python
encoded_df = inspector.extract_normalized_categorical_data(
    method="onehot"
)
```

### Ordinal Encoding

```python
encoded_df = inspector.extract_normalized_categorical_data(
    method="ordinal"
)
```

### Uniform Encoding

```python
encoded_df = inspector.extract_normalized_categorical_data(
    method="uniform"
)
```

---

## Create a Machine Learning Ready Dataset

Merge normalized numerical features and encoded categorical features.

```python
final_df = inspector.create_normalized_data_df(
    numeric_method="robust",
    categorical_method="onehot"
)
```

---

# Exploratory Data Analysis (Visual)

## Numerical Visualization

Generate a combined statistical view for numerical variables.

```python
inspector.plot_numerical([
    "Age",
    "Fare"
])
```

Each numerical column produces:

### Violin Plot

Shows:

* Distribution shape
* Median
* Density
* Potential outliers

### Scatter Plot

Plots:

```text
Index vs Variable Value
```

Useful for:

* Trend detection
* Outlier identification
* Data consistency checks

### Histogram

Shows:

* Frequency distribution
* Skewness
* Data spread

---

## Categorical Visualization

```python
inspector.plot_categorical([
    "Sex",
    "Embarked"
])
```

Creates frequency bar charts displaying:

* Category counts
* Percentage labels

---

## Relationship Analysis

```python
inspector.plot_relationship(
    "Sex",
    "Fare"
)
```

The method automatically chooses the correct visualization.

### Numerical vs Numerical

```python
inspector.plot_relationship(
    "Age",
    "Fare"
)
```

Creates:

* Scatter Plot
* OLS Trendline

---

### Categorical vs Numerical

```python
inspector.plot_relationship(
    "Sex",
    "Fare"
)
```

Creates:

* Box Plot
* Individual Data Points

---

### Categorical vs Categorical

```python
inspector.plot_relationship(
    "Sex",
    "Embarked"
)
```

Creates:

* Grouped Bar Chart

---

# Correlation Analysis

## Numerical Correlation Heatmap

Uses Pearson Correlation Coefficients.

```python
inspector.plot_numerical_correlation()
```

### Interpretation

| Correlation | Meaning          |
| ----------- | ---------------- |
| +1          | Perfect Positive |
| +0.7 to +1  | Strong Positive  |
| 0           | No Relationship  |
| -0.7 to -1  | Strong Negative  |
| -1          | Perfect Negative |

---

## Categorical Correlation Heatmap

Uses Cramér's V.

```python
inspector.plot_categorical_correlation()
```

### Interpretation

| Cramér's V | Strength   |
| ---------- | ---------- |
| 0.00–0.10  | Negligible |
| 0.10–0.30  | Weak       |
| 0.30–0.50  | Moderate   |
| >0.50      | Strong     |

---

## Numerical-Categorical Association Analysis

```python
inspector.correlate_num_to_cat()
```

Uses:

* ANOVA F-Statistic
* Point-Biserial Correlation

to evaluate relationships between numerical and categorical variables.

---

## Unified Association Heatmap

```python
inspector.plot_all_associations_heatmap()
```

Combines:

* Pearson Correlation
* Cramér's V
* Numerical-Categorical Associations

into a single relationship matrix.

---

# Retrieve Current Dataset

```python
df = inspector.get_dataframe()
```

---

# Complete Example

```python
from datainspector import DataInspector

inspector = DataInspector()

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

inspector.upload_data(url)

inspector.clean_garbage_values()

inspector.auto_convert_types()

inspector.data_summary()

inspector.handle_missing_values(
    strategy="median",
    columns=["Age"]
)

inspector.remove_duplicates()

inspector.handle_outliers(
    column="Fare",
    remove=True
)

final_df = inspector.create_normalized_data_df(
    numeric_method="robust",
    categorical_method="onehot"
)

inspector.plot_numerical([
    "Age",
    "Fare"
])

inspector.plot_numerical_correlation()
```

---

# Author

**Harendra Gunawardana**