# Student Academic Performance Prediction: KNN vs Naive Bayes

> Early-warning system that predicts Low / Medium / High academic performance using KNN and Naive Bayes.

## Aim
To build and compare K-Nearest Neighbours (KNN) and Naive Bayes classifiers that predict semester GPA performance bands (Low / Medium / High) from attendance and assignment patterns, and to deliver the results as an early-warning dashboard for mentors.

## Problem Statement
Many students fall behind academically before anyone notices. This project uses institutional LMS records to identify at-risk students early, so mentors can step in on time.

## Features
- Data cleaning: duplicate removal, missing value handling, outlier capping
- GPA binned into **Low / Medium / High** performance bands
- KNN and Gaussian Naive Bayes models with feature scaling
- 5-fold cross-validation and KNN `k` tuning
- Accuracy, classification report, and confusion matrix for both models
- Early-warning output: at-risk students ranked by risk probability

## Dataset
Expected file: `lms_records.csv`

| Column | Description |
|---|---|
| attendance_pct | Attendance percentage |
| assignments_submitted_pct | Percentage of assignments submitted |
| avg_assignment_score | Average assignment score |
| late_submissions | Number of late submissions |
| lms_logins_per_week | Weekly LMS logins |
| semester_gpa | Semester GPA (target) |

If `lms_records.csv` is not found, the script generates synthetic demo data automatically.

## Tech Stack
Python, pandas, NumPy, scikit-learn, Matplotlib

## Project Structure
```
├── student_performance_knn_nb.py   # main script
├── lms_records.csv                 # dataset (optional)
├── mentor_alerts.csv               # at-risk student list
├── mentor_dashboard.png            # result charts
└── README.md
```

## Installation
```bash
git clone https://github.com/Fathima1616/student-performance-knn-naive-bayes.git
cd student-performance-knn-naive-bayes
pip install pandas numpy scikit-learn matplotlib
```

## Usage
```bash
python student_performance_knn_nb.py
```

## Methodology
1. Load LMS records
2. Clean and transform data (duplicates, missing values, outliers)
3. Convert GPA into three performance bands
4. Split data 80/20 (stratified)
5. Train KNN and Naive Bayes with standard scaling
6. Evaluate using cross-validation and test metrics
7. Generate the mentor early-warning report and dashboard

## Output
- `mentor_alerts.csv`: students flagged as Low performers, ranked by risk %
- `mentor_dashboard.png`: model accuracy comparison, KNN k-tuning curve, risk distribution

## Results
| Model | CV Accuracy | Test Accuracy |
|---|---|---|
| KNN (k=7) | 0.778 | 0.770 |
| Naive Bayes | 0.790 | 0.830 |

**Best model:** Naive Bayes (83% test accuracy)
**Best k for KNN:** 17 (from cross-validation tuning)
**At-risk students flagged:** 28 out of 100 test students

## Observations
- Naive Bayes beat KNN on both CV accuracy (0.790 vs 0.778) and test accuracy (0.830 vs 0.770).
- **High** and **Low** bands are predicted well (F1 about 0.83 to 0.89). The **Medium** band is hardest (F1 0.66 to 0.74) because it overlaps with both neighbouring groups.
- Naive Bayes recalled High students almost perfectly (0.97).

## Dashboard
![Mentor Dashboard](mentor_dashboard.png)

## Future Work
- Live dashboard using Streamlit
- Try more models (Random Forest, SVM)
- Tune hyperparameters with GridSearchCV
- Integrate directly with the institutional LMS
- Validate with real LMS data from the institution

## Author
**Shaik Fathima Bee** (23ME1A5460)
Guide: Mr. R. Sri Ramu
Course: Machine Learning & Predictive Analytics
