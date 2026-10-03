# Data Cleaning & Visualization Project

## Project Overview

This project focuses on cleaning, preprocessing, analyzing, and visualizing a raw dataset using Python.

The project demonstrates practical data analysis techniques including missing value handling, duplicate detection, outlier detection, exploratory data analysis, and data visualization.

## Objectives

* Clean raw data
* Handle missing values
* Detect and remove duplicate records
* Detect numerical outliers using the IQR method
* Perform exploratory data analysis
* Create meaningful visualizations
* Generate key insights from the dataset
* Export the cleaned dataset

## Technologies Used

* Python
* Pandas
* Matplotlib
* Seaborn

## Data Cleaning

The following preprocessing steps were performed:

1. Dataset loading
2. Data understanding
3. Missing value detection and handling
4. Duplicate detection and removal
5. Outlier detection using IQR
6. Cleaned dataset export

## Visualizations

The project generates the following visualizations:

* Salary Distribution
* Average Salary by Department
* Department Distribution
* Age vs Salary
* Salary Outlier Detection
* Correlation Heatmap

## Project Structure

```text
data-cleaning-visualization/
│
├── data.py
├── data.csv
├── cleaned_data.csv
├── README.md
├── .gitignore
│
├── 01_salary_distribution.png
├── 02_average_salary_department.png
├── 03_department_distribution.png
├── 04_age_vs_salary.png
├── 05_salary_outliers.png
└── 06_correlation_heatmap.png
```

## How to Run

Install the required libraries:

```bash
py -m pip install pandas matplotlib seaborn
```

Run the project:

```bash
py data.py
```

## Output

The project produces a cleaned CSV file and six visualization images that help identify patterns, distributions, relationships, and outliers in the dataset.

## Key Learning Outcomes

* Data preprocessing using Pandas
* Missing value handling
* Duplicate detection
* Outlier detection using IQR
* Exploratory Data Analysis
* Data visualization
* Data storytelling
* Exporting cleaned datasets and charts
