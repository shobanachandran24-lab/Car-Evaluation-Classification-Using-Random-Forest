# Car Evaluation Classification Using Random Forest

A machine learning project that predicts the **acceptability of a car** (`unacc`, `acc`, `good`, `vgood`) from its price, comfort, and safety attributes, using a **Random Forest Classifier**.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Results](#results)
- [Installation and Usage](#installation-and-usage)
- [Future Improvements](#future-improvements)
- [Acknowledgements](#acknowledgements)

---

## Overview

The goal of this project is to classify cars into one of four acceptability levels based on six categorical attributes. The model is built with Python and scikit-learn, and it reaches about **95% accuracy** on the test set.

## Dataset

- **Source:** [UCI Machine Learning Repository: Car Evaluation](https://archive.ics.uci.edu/dataset/19/car+evaluation)
- **Creator:** Marko Bohanec
- **Instances:** 1,728
- **Features:** 6 (all categorical)
- **Missing values:** None
- **File used:** `car.data` (comma-separated, no header)

### Features

| Feature    | Description          | Values                        |
|------------|----------------------|-------------------------------|
| `buying`   | Buying price         | vhigh, high, med, low         |
| `maint`    | Maintenance cost     | vhigh, high, med, low         |
| `doors`    | Number of doors      | 2, 3, 4, 5more                |
| `persons`  | Passenger capacity   | 2, 4, more                    |
| `lug_boot` | Luggage boot size    | small, med, big               |
| `safety`   | Estimated safety     | low, med, high                |

### Target (`class`)

| Class   | Meaning                  | Approx. share |
|---------|--------------------------|---------------|
| `unacc` | Unacceptable             |  70%          |
| `acc`   | Acceptable               |  22%          |
| `good`  | Good                     |  4%           |
| `vgood` | Very good                |  4%           |

> The classes are imbalanced, which affects performance on `good` and `vgood`.

## Project Structure

```
├── car.data              # Dataset from the UCI repository
├── Random_forest.ipynb   # Jupyter notebook with the full analysis
└── README.md             # Project documentation
```

## Workflow

1. **Load data**: Read `car.data` with Pandas and assign column names.
2. **Check data quality**: Confirm there are no missing values.
3. **Split features and target**: `X` = six attributes, `y` = `class`.
4. **Encode features**: Apply one-hot encoding with `pd.get_dummies()`.
5. **Train/test split**: 80% training and 20% testing (`random_state=42`).
6. **Train model**: `RandomForestClassifier(n_estimators=100, random_state=42)`.
7. **Evaluate**: Accuracy, classification report, and confusion matrix heatmap.

## Results

| Metric           | Value   |
|------------------|---------|
| Test accuracy    | 95.4%   |
| Test samples     | 346     |

**Per-class performance**

| Class   | Precision | Recall | F1-score |
|---------|-----------|--------|----------|
| `acc`   | 0.97      | 0.84   | 0.90     |
| `good`  | 0.59      | 0.91   | 0.71     |
| `unacc` | 0.98      | 1.00   | 0.99     |
| `vgood` | 0.83      | 0.88   | 0.86     |

**Observations**

- The model is very strong on the majority class `unacc`.
- Most errors are `acc` cars predicted as `good`.
- The minority classes (`good`, `vgood`) are harder to predict because there are fewer samples.

## Installation and Usage

**Requirements**

- Python 3.8+
- pandas
- scikit-learn
- seaborn
- matplotlib
- jupyter

**Install dependencies**

```bash
pip install pandas scikit-learn seaborn matplotlib jupyter
```

**Run the project**

```bash
git clone <your-repo-url>
cd <your-repo-folder>
jupyter notebook Random_forest.ipynb
```

Make sure `car.data` is in the same folder as the notebook.

## Future Improvements

- Handle class imbalance (class weights or SMOTE)
- Tune hyperparameters with `GridSearchCV`
- Analyze feature importance
- Compare with other models (Decision Tree, SVM, Gradient Boosting)
- Use cross-validation for more reliable evaluation

## Acknowledgements

- Dataset: Bohanec, M. (1997). *Car Evaluation* [Dataset]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/19/car+evaluation
- Libraries: scikit-learn, Pandas, Seaborn, Matplotlib
