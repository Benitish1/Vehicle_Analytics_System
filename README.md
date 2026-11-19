# Vehicle Analytics System 

## Overview

The **Vehicle Analytics System** is a Python-based data analytics project designed to process and analyze vehicle-related datasets using machine learning techniques. The system performs **classification, clustering, and regression analysis**, and displays the results through a frontend interface for easy visualization and interpretation.

This project demonstrates how backend data processing in Python can be integrated with a frontend dashboard to present insights in a user-friendly way.

## 🚀 Live Demo

**Try the application:** [Vehicle Analytics System](https://vehicle-analytics-system.vercel.app)

---

## Features

* **Machine Learning Models**

  * Classification models for predicting categories.
  * Clustering models for grouping similar data points.
  * Regression models for predicting numerical values.

* **Data Visualization**

  * Results displayed on a frontend interface.
  * Geographic visualization using Rwanda district data.

* **Geospatial Support**

  * Includes `rwanda_districts.geojson` for mapping and geographic data analysis.

* **Modular Model Generators**

  * Organized model training scripts for different machine learning tasks.

---

## Project Structure

```
Vehicle_Analytics_System/
│
├── config/                     # Django project configuration
│
├── dummy-data/                 # Sample datasets used for testing
│
├── model_generators/           # Machine learning model training modules
│   ├── classification/         # Classification model scripts
│   ├── clustering/             # Clustering model scripts
│   └── regression/             # Regression model scripts
│
├── predictor/                  # Prediction logic and API endpoints
│
├── venv/                       # Python virtual environment
│
├── Rwanda_Districts.geojson    # Rwanda geographic district data
│
├── clustering_results.json     # Stored clustering results
│
├── db.sqlite3                  # SQLite database
│
├── manage.py                   # Django project manager
│
├── requirements.txt            # Python dependencies
│
└── README.md                   # Project documentation
```

---

## Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **GeoJSON**
* **Frontend technologies (HTML / CSS / JavaScript)**

---

## Machine Learning Components

### Classification

Used to categorize data into predefined classes.

Example tasks may include:

* Vehicle type prediction
* Traffic category detection

---

### Clustering

Groups similar data points without predefined labels.

Example uses:

* Traffic pattern grouping
* Vehicle behavior segmentation

---

### Regression

Predicts continuous numerical values.

Example uses:

* Vehicle counts prediction
* Traffic flow estimation

---

## Geospatial Data

The project uses **`rwanda_districts.geojson`** to enable geographic visualization and mapping of analytics results across Rwanda's districts.

This allows the system to:

* Display analytics results geographically
* Support map-based dashboards

---

## Installation

### 1. Clone the repository

After cloning;
```bash
cd Vehicle_Analytics_System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the project

Run the main script or start the backend service depending on the project setup.

```bash
python manage.py runserver
```

---

## Usage

1. Load the vehicle dataset.
2. Train or run the machine learning models.
3. Generate predictions or clustering results.
4. Display analytics results on the frontend dashboard.

---

## Future Improvements

* Add real-time traffic data integration
* Improve dashboard visualization
* Deploy the system to a cloud platform
* Expand machine learning model accuracy
