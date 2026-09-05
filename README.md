# Food Ordering Behavior Analysis & KNN Classification

A Data Science project evaluating K-Nearest Neighbors (KNN) classification models on a food ordering behavior dataset. The analysis evaluates KNN performance across various target variables (`rainy_weather`, `is_repeat_order`, and `hunger_level`) to determine predictive modeling effectiveness.

---

## Table of Contents
- [Overview](#overview)
- [Dataset Details](#dataset-details)
- [Project Workflow](#project-workflow)
- [Model Evaluation & Results](#model-evaluation--results)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)

---

## Overview

This project analyzes customer food ordering behavior across 50,000 transactions. It explores whether K-Nearest Neighbors (KNN) classification algorithms can reliably predict target attributes like weather conditions, repeat order status, or customer hunger levels based on order demographics and behavioral metrics.

---

## Dataset Details

The dataset contains **50,000 rows** and **19 columns** with no missing values or duplicate records.

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `order_id` | Integer | Unique identifier for the order |
| `user_id` | Integer | Unique identifier for the user |
| `age` | Integer | Age of the customer |
| `city` | Object | Delivery city (e.g., Pune, Mumbai, Delhi, Chandigarh, Hyderabad) |
| `order_time` | Object | Time slot of the order (e.g., Morning, Afternoon, Evening, Night) |
| `day_type` | Object | Weekday vs. Weekend |
| `cuisine` | Object | Type of food (e.g., Chinese, South Indian, Biryani, Fast Food, Desserts) |
| `meal_type` | Object | Meal category (e.g., Breakfast, Lunch, Dinner, Snacks) |
| `restaurant_type` | Object | Restaurant tier (e.g., Budget, Mid-range, Premium) |
| `order_value` | Integer | Total transaction value |
| `discount_applied` | Object | Whether a discount was used (`Yes`/`No`) |
| `delivery_fee` | Integer | Fee charged for delivery |
| `time_taken_to_order` | Integer | Time spent placing the order (in minutes) |
| `rating_given` | Integer | Customer rating given for the order |
| `is_repeat_order` | Object | Indicates if the order is a repeat (`Yes`/`No`) |
| `mood` | Object | Customer mood (e.g., Celebrating, Lazy, Happy, Stressed) |
| `hunger_level` | Object | Self-reported hunger level (`Low`, `Medium`, `High`) |
| `company` | Object | Dining company (e.g., Partner, Family, Friends, Alone) |
| `rainy_weather` | Object | Indicates whether weather was rainy (`Yes`/`No`) |

---

## Project Workflow

1. **Data Preprocessing & EDA:**
   - Loaded and inspected data (`df.info()`, `df.isna().sum()`, `df.duplicated().sum()`).
   - Dropped non-predictive identifiers (`order_id`, `user_id`).
   - Applied One-Hot Encoding (`pd.get_dummies`) to categorical features with `drop_first=True`.

2. **Train-Test Split & Scaling:**
   - Stratified train-test split (80% training, 20% testing).
   - Scaled feature sets using `StandardScaler`.

3. **KNN Classification & Hyperparameter Tuning:**
   - Tested multiple K values (K in [3, 5, 7, 10, 11, 13, 15, 17, 20]) across three distinct targets:
     - **Target 1:** `rainy_weather` (Binary Classification)
     - **Target 2:** `is_repeat_order` (Binary Classification)
     - **Target 3:** `hunger_level` (Multi-class Classification)

---

## Model Evaluation & Results

### Performance Summary

| Target Variable | Target Type | Tested K Range | Best K | Peak Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **`rainy_weather`** | Binary | 3 – 20 | **10** | **50.70%** |
| **`is_repeat_order`** | Binary | 3 – 20 | **13** | **50.40%** |
| **`hunger_level`** | Multi-class | 3 – 20 | **20** | **34.63%** |

### Final Model Selection

Among all experiments, the **Rainy Weather Prediction Model** yielded the highest accuracy.

* **Selected Algorithm:** K-Nearest Neighbors (KNN)
* **Target Variable:** `rainy_weather`
* **Optimal K Value:** 10
* **Accuracy Score:** **50.70%**

> **Note:** The performance across targets hovers around baseline/random-guess thresholds (~50% for binary, ~33% for 3-class), indicating that feature distributions in this dataset lack strong spatial clustering for distance-based algorithms like KNN.

---

## Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/mohitbatheja/food-ordering-knn-analysis.git](https://github.com/your-username/food-ordering-knn-analysis.git)
   cd food-ordering-knn-analysis


```


3. **Install Dependencies:**
```bash
pip install pandas scikit-learn

```



---

## Usage

1. Place `food_ordering_behavior_dataset.csv` into the project root directory.
2. Open and run the Jupyter Notebook:
```bash
jupyter notebook

```


3. Run all notebook cells sequentially to execute data preprocessing, model fitting, and metrics printing.

```

```
