import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. DATA CREATE PANROM ---
data = {
    'Student_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 12],
    'Name': ['Asha', 'Ravi', 'Meena', 'Kumar', 'Priya', 'Arun', 'Divya', 'Suresh', 'Lakshmi', 'Vijay', 'Vijay', 'Anitha'],
    'Department': ['CSE', 'ECE', 'CSE', 'MECH', 'ECE', 'CSE', 'MECH', 'CSE', 'ECE', 'MECH', 'MECH', 'CSE'],
    'Age': [20, 21, 19, 22, 20, None, 21, 20, 19, 23, 23, 20],
    'Gender': ['Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female'],
    'Exam_Score': [85, 72, 90, 65, None, 78, 88, 55, 92, 70, 70, 81],
    'Attendance': [90, 85, 95, 75, 88, 82, 91, 60, 96, 78, 78, 89]
}
df = pd.DataFrame(data)

print("--- ORIGINAL DATASET ---")
print(df.head(10))
print("Original Dataset Shape:", df.shape)

# --- 2. DUPLICATE REMOVE PANROM ---
print("\nDuplicate Records:", df.duplicated().sum())
df = df.drop_duplicates()
print("After Removing Duplicates Shape:", df.shape)

# --- 3. MISSING VALUES HANDLE PANROM ---
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# --- 4. CLASSIFY STUDENT PERFORMANCE ---
def classify_student_performance(score):
    if score >= 85:
        return 'Excellent'
    elif score >= 70:
        return 'Good'
    elif score >= 50:
        return 'Average'
    else:
        return 'Needs Improvement'

df['Performance'] = df['Exam_Score'].apply(classify_student_performance)

# --- 5. CLEANED DATASET ---
print("\n--- CLEANED DATASET ---")
print(df.head(10))
print("Cleaned Dataset Shape:", df.shape)

# --- 6. STATISTICAL SUMMARY ---
print("\n--- STATISTICAL SUMMARY ---")
print(df.describe().round(2))

# --- 7. DEPARTMENT-WISE AVERAGE ---
print("\n--- DEPARTMENT-WISE AVERAGE ---")
dept_analysis = df.groupby('Department')['Exam_Score'].mean()
print(dept_analysis.round(2))

# --- 8. PERFORMANCE DISTRIBUTION ---
print("\n--- PERFORMANCE DISTRIBUTION ---")
print(df['Performance'].value_counts())

# --- 9. GRAPH 1 - DEPARTMENT WISE ---
plt.figure(figsize=(8, 5))
plt.bar(dept_analysis.index, dept_analysis.values, color='skyblue')
plt.title('Department-wise Average Exam Score')
plt.xlabel('Department')
plt.ylabel('Average Exam Score')
plt.show()

# --- 10. GRAPH 2 - PERFORMANCE DISTRIBUTION ---
perf_counts = df['Performance'].value_counts()
plt.figure(figsize=(8, 5))
plt.bar(perf_counts.index, perf_counts.values, color='lightgreen')
plt.title('Student Performance Distribution')
plt.xlabel('Performance')
plt.ylabel('Count')
plt.show()

print("\nData Cleaning and EDA Completed Successfully!")
