<div align="center">
  <img src="images/Auto_EDA.png" alt="auto_eda_logo"/>
</div>


## Introduction

Throughout my data science journey, I have learned that it is a good practice to understand the data first and try to gather as many insights from it. Exploratory Data Analysis (EDA) is all about making sense of data in hand, before getting dirty with machine learning and sophiticated algorithm. 

While there are plenty of Python libraries that can help create beautiful and complex visualizations, I often find myself starting with the most simplistic analyses: count plot, histogram, scatter plot, boxplot, etc. This initial EDA workflow is very similar for each new data set. But unfortunately, they are tedious. Converting to correct data types, selecting the right variable type for the right plot, itterate through all possible variable combinations, adjusting plot asthetic and labels, etc. These are the tasks I would love to do... once. As someone that does not find great joy in completing repetitive tasks, I set out to build a tool that allow me to me as lazy as possible.

## Description
Auto_EDA is a python library that automates common tasks in your exploratory data analysis. This includes missing values visualization, missing values handling, variable types handling, predictive modeling and a variety of univariate and bivariate graphs. The goal is to provide a fast and effective tool for discovering insights, so you can quickly move on machine learning model.

## Installation

:warning: Auto_EDA is only compatible with **Python 3**.

:warning: Decision Tree visualizer requires [graphviz](https://github.com/parrt/dtreeviz).

**Install Via GitHub**

```sh
> pip install git+https://github.com/GrandPurpleOcelot/Auto-EDA 
```
## Usage

```python
from auto_eda import *;
```

## Table of Contents

- [Table of contents](#table-of-contents)
- [Installation](#installation)
- [Dataset Overview](#dataset-overview)

### Descriptive Statistics

####  Dataset Overview

1. Number of variables

2. Number of observations

3. Memory usage

### Missing Values

1. Visualize missing values using heatmap

2. Suggestions for handling missing value

### Variable Types

1. Print out current data types from Pandas

2. Type conversion suggestions: 

 * String datetime -> datetime
 
 * Small interger (for example: boolean)  -> categorical type
 
 * String float -> float
 
 * Maximum cardinality (number of unique == number of observations) -> remove

### Visualization

#### Univariate plots

1. Histogram (for numerical data)

2. Count plot (for categorical data)

3. World cloud (for text data)

#### Bivariate plots 

User can specify a categorical column for grouping. 

1. Correlation plots (heat map and scatter plot): for numerical and numerical data
    * Principal Component Analysis

2. Box plots: for numerical and categorical data

3. Relative frequency plots: for categorical and categorical data
    * Correspondence Analysis

4. Trend plot (connected scatterplot): if timeseries data is present


### Statistical Modeling

User can specify a target variable for classification/regression task using Classification And Regression Tree (CART).

1. Classification Report (train on 75% of data, test on 25%)

2. Bar chart of relative feature importance

3. Decision tree visualization with Dtreeviz

