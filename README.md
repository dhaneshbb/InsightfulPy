# insightfulpy

**insightfulpy** is a comprehensive Python package designed to simplify Exploratory Data Analysis (EDA) workflows. It provides powerful utilities for analyzing both numerical and categorical data, detecting outliers, handling missing values, and generating insightful visualizations.

---

## Features

The provided code is an **exploratory data analysis (EDA) toolkit** that includes various functions for analyzing and visualizing both categorical and numerical data. Below are the key features:

1. **Categorical Data Analysis**  
   - Computes summary statistics such as unique values, mode, missing percentage, and frequency distribution.  
   - Identifies high-cardinality categorical variables.  
   - Provides bar charts, pie charts, and heatmaps for categorical relationships.  

2. **Numerical Data Analysis**  
   - Generates statistical summaries including mean, median, standard deviation, skewness, and kurtosis.  
   - Performs normality tests (Shapiro-Wilk, Kolmogorov-Smirnov).  
   - Detects outliers using the IQR method and identifies interconnected outliers.  
   - Supports box plots, KDE plots, and scatter plots for numerical relationships.  

3. **Visualization and Batch Processing**  
   - Visualizes missing values using a missing value matrix and bar chart.  
   - Batch-wise KDE plots, box plots, scatter plots, and QQ plots.  
   - Numerical vs categorical visualizations using box and violin plots.  

4. **Data Integrity and Data Quality Checks**  
   - Detects missing and infinite values.  
   - Identifies mixed data types in columns.  
   - Compares column profiles across multiple datasets.  

5. **Linked Data Analysis**  
   - Identifies common key columns across datasets.  
   - Analyzes interconnected outliers affecting multiple columns.  
   - Compares shared columns across datasets for consistency.  

The toolkit provides a structured and efficient approach to EDA, enabling automated data profiling, anomaly detection, and visualization for better data-driven insights.

---

## Installation

```bash
pip install insightfulpy
```

Or, if you're installing directly from the repository:

```bash
pip install git+https://github.com/dhaneshbb/insightfulpy.git
```

---

## Dependencies

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `researchpy`
- `tableone`
- `missingno`
- `scipy`
- `tabulate`

All dependencies are automatically installed with the package.

---

## Usage

### Importing the Package

```python
from insightfulpy.eda import *
```

---

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

---

## License

This project is licensed under the MIT License. See the (LICENSE) file for details.

---

## Acknowledgements

- Inspired by best practices in EDA and data visualization.
- Thanks to the open-source community for the amazing tools and libraries!

