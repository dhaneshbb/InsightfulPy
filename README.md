# InsightfulPy

**InsightfulPy** is a comprehensive Python package designed to simplify Exploratory Data Analysis (EDA) workflows. It provides powerful utilities for analyzing both numerical and categorical data, detecting outliers, handling missing values, and generating insightful visualizations.

---

## Features

-  **Numerical Analysis**: Summarize numerical features with detailed statistics like mean, median, mode, skewness, kurtosis, and more.
-  **Categorical Analysis**: Generate frequency tables, detect high-cardinality features, and analyze mixed data types.
-  **Visualization Tools**: Create box plots, KDE plots, QQ plots, scatter plots, bar charts, and pie charts effortlessly.
-  **Outlier Detection**: Identify outliers using the IQR method and visualize them.
-  **Missing Data Handling**: Visualize missing data patterns using `missingno`.
-  **Customizable Summaries**: Grouped statistical summaries with `TableOne` for deeper insights.

---

## Installation

```bash
pip install InsightfulPy
```

Or, if you're installing directly from the repository:

```bash
pip install git+https://github.com/dhaneshbb/InsightfulPy.git
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
import pandas as pd
from InsightfulPy.eda import *
from InsightfulPy.utils import *
```

---

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author

- **Your Name**  
  [GitHub](https://github.com/dhaneshbb/InsightfulPy) | [Email](dhaneshbb5@gmail.com)

---

## Acknowledgements

- Inspired by best practices in EDA and data visualization.
- Thanks to the open-source community for the amazing tools and libraries!

