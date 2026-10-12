# Student Academic Performance Prediction: KNN vs Naive Bayes

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

## Installation
```bash
git clone https://github.com/<your-username>/student-performance-knn-naive-bayes.git
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
| KNN | add your value | add your value |
| Naive Bayes | add your value | add your value |

## Future Work
- Live dashboard using
