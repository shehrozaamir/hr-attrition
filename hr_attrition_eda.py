# ============================================================
#  IBM HR Analytics - Employee Attrition
#  Step 1: Data Loading + Basic EDA + Cleaning
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. LOAD DATA ─────────────────────────────────────────────
# utf-8-sig: CSV ke pehle column mein BOM hota hai, isse 'Age' sahi naam se aata hai
df = pd.read_csv('HR-Employe-Attrition.csv', encoding='utf-8-sig')

print("=" * 50)
print("  FIRST 5 ROWS (df.head())")
print("=" * 50)
print(df.head())


# ── 2. BASIC INFO ────────────────────────────────────────────
print("\n" + "=" * 50)
print("  SHAPE  (rows, columns)")
print("=" * 50)
print(df.shape)
# Output: (1470, 35)
# 1470 employees, 35 features/columns


print("\n" + "=" * 50)
print("  DATA TYPES (df.info())")
print("=" * 50)
print(df.info())
# object  = text / categorical column  (jaise 'Yes'/'No', 'Male'/'Female')
# int64   = numeric / number column    (jaise Age, MonthlyIncome)


# ── 3. MISSING VALUES ────────────────────────────────────────
print("\n" + "=" * 50)
print("  MISSING VALUES (df.isnull().sum())")
print("=" * 50)
missing = df.isnull().sum()
print(missing)
print(f"\nTotal missing cells: {missing.sum()}")
# Agar 0 aaye → dataset bilkul clean hai, koi missing value nahi!


# ── 4. TARGET COLUMN CHECK (Attrition) ───────────────────────
print("\n" + "=" * 50)
print("  TARGET COLUMN: Attrition")
print("=" * 50)

# Unique values dekho (Yes ya No)
print("Unique values :", df['Attrition'].unique())

# Kitni baar 'Yes' aur kitni baar 'No'
print("\nValue Counts:")
print(df['Attrition'].value_counts())

# ── 5. VISUALIZE TARGET BALANCE ──────────────────────────────
# Yeh chart dikhayega ke dataset balanced hai ya imbalanced
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='Attrition', data=df,
                   palette={'No': '#2ecc71', 'Yes': '#e74c3c'})

# Har bar ke upar count likho
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('Attrition Distribution\n(Target Column Balance Check)', fontsize=13)
plt.xlabel('Attrition (No = Stay  |  Yes = Left)')
plt.ylabel('Number of Employees')
plt.tight_layout()
plt.savefig('01_attrition_distribution.png', dpi=150)
plt.show()
print("\n[Chart saved as: 01_attrition_distribution.png]")

# ── SUMMARY ──────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  QUICK SUMMARY")
print("=" * 50)
print(f"Total Employees   : {len(df)}")
print(f"Total Features    : {df.shape[1]}")
print(f"Employees Stayed  : {(df['Attrition'] == 'No').sum()}  ({(df['Attrition']=='No').mean()*100:.1f}%)")
print(f"Employees Left    : {(df['Attrition'] == 'Yes').sum()}  ({(df['Attrition']=='Yes').mean()*100:.1f}%)")
print(f"Missing Values    : {df.isnull().sum().sum()}")
print(f"Numeric Columns   : {df.select_dtypes(include='number').shape[1]}")
print(f"Categorical Cols  : {df.select_dtypes(include='object').shape[1]}")
print("=" * 50)
print("\n✅ Step 1 Complete! Agle step mein EDA aur Visualization karenge.")


# ── 6. REMOVE DUPLICATES ─────────────────────────────────────
df = df.drop_duplicates()   # is dataset mein 0 duplicates hain


# ── 7. OUTLIERS: DELETE NAHI KARNE ───────────────────────────
# IQR ke hisaab se kai columns mein outliers hain (MonthlyIncome 114,
# YearsSinceLastPromotion 107, YearsAtCompany 104, TotalWorkingYears 63),
# lekin ye sab valid values hain (right-skewed tail), koi error nahi.
# Isliye rows delete nahi, log transform lagate hain.
#
# StockOptionLevel / PerformanceRating / TrainingTimesLastYear discrete
# columns hain, in pe IQR outlier removal meaningful nahi.

print("\nSkew before log1p:")
skewed_cols = ['MonthlyIncome', 'TotalWorkingYears',
               'YearsAtCompany', 'YearsSinceLastPromotion']
print(df[skewed_cols].skew().round(2))

for c in skewed_cols:
    df[c] = np.log1p(df[c])

print("\nSkew after log1p:")
print(df[skewed_cols].skew().round(2))


# ── 8. USELESS COLUMNS DROP ──────────────────────────────────
# EmployeeCount, StandardHours, Over18 = constant (sab rows mein same value)
# EmployeeNumber = sirf ID
df = df.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'])
print("\nShape after cleaning:", df.shape)   # (1470, 31)


# ── 9. BOXPLOTS (transform ke baad) ──────────────────────────
numerical_cols = df.select_dtypes(include=np.number).columns

# for i in numerical_cols:
#     sns.boxplot(x=df[i])
#     plt.show()


# Features, Target & Train-Test Split
X = df.drop('Attrition', axis=1)
y = df['Attrition'].map({'Yes': 1, 'No': 0})
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
numeric_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate',
                'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
                'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear',
                'YearsAtCompany', 'YearsInCurrentRole',
                'YearsSinceLastPromotion', 'YearsWithCurrManager']

# Numeric dikhte hain par asal mein levels/ratings hain: as-is rakho, one-hot nahi
ordinal_cols = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement',
                'JobLevel', 'JobSatisfaction', 'PerformanceRating',
                'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance']

# Text columns: inpe one-hot encoding
categorical_cols = ['BusinessTravel', 'Department', 'EducationField',
                    'Gender', 'JobRole', 'MaritalStatus', 'OverTime']

#Handle class imbalance
neg, pos = np.bincount(y_train)
scale_weigths = neg/pos




#Preprocessing for Logistic Regression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
numeric_cols_all = numeric_cols + ordinal_cols

preprocessor_lr = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_cols_all),

        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy="constant", fill_value="Missing")),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ])




# Preprocessing for Randomforest
preprocessor_rf = ColumnTransformer(
    transformers=[
        #Pipeline - for numerical cols
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median'))
        ]), numeric_cols),

        #Pipeline - for Categorical cols.
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy="constant", fill_value="Missing")),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ])


#Stratified Cross-Validation
from sklearn.model_selection import StratifiedKFold, cross_validate
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {'roc_auc':'roc_auc', 'accuracy':'accuracy', 'precision':'precision', 'recall':'recall', 'f1':'f1'}


#Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_cv_pipeline = Pipeline([
    ('preprocessor', preprocessor_lr),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])


#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_cv_pipeline = Pipeline([
    ('preprocessor', preprocessor_rf),
    ('classifier', RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        class_weight='balanced',
        random_state=42
    ))
])
for cv_name, pipe in [("Logistic Regression", lr_cv_pipeline),
                      ("Random Forest", rf_cv_pipeline)]:

                      cv_results = cross_validate(pipe, X_train , y_train, cv=cv, scoring=scoring, n_jobs=-1)

                      print(cv_name)
                      print(f"ROC-AUC : {cv_results['test_roc_auc'].mean()}")
                      print(f"Accuracy : {cv_results['test_accuracy'].mean()}")
                      print(f"Precision : {cv_results['test_precision'].mean()}")
                      print(f"Recall : {cv_results['test_recall'].mean()}")
                      print(f"F1 : {cv_results['test_f1'].mean()}")
                      print()



#Baseline Model — Logistic Regression
from sklearn.pipeline import Pipeline
baseline_model = Pipeline([
    ('preprocessor', preprocessor_lr),
    ('classifier', LogisticRegression(class_weight= "balanced"))
])
baseline_model.fit(X_train, y_train)

# Evaluation Helper Function
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, precision_recall_curve
def evaluate_model(model_name, model, X_test, y_test, threshold=None):
  y_pred_prob = model.predict_proba(X_test)[:,1]

  if threshold:
    y_pred = (y_pred_prob > threshold).astype(int)
  else:
    y_pred = model.predict(X_test)

  #Metrics
  accuracy     = accuracy_score(y_test, y_pred)    #Overall model's preformance
  precision    = precision_score(y_test, y_pred)   #When the model says 1, how oftenly is it actually 1.
  recall       = recall_score(y_test, y_pred)      #of the actual 1 how many did the model catches
  f1           = f1_score(y_test, y_pred)          #Harmonic mean of precision and recall

  conf_matrix   = confusion_matrix(y_test, y_pred)
  class_report  = classification_report(y_test, y_pred)


  print(f"{model_name} Metrics:")
  if threshold is not None: # Only print threshold if provided
    print(f"Used Threshold : {threshold:.2f}")
  print(f"Accuracy : {accuracy:.2f}")
  print(f"Precision : {precision:.2f}")
  print(f"Recall : {recall:.2f}")
  print(f"F1 : {f1:.2f}")
  print()
  print(f"{model_name} Classification Report:")
  print(class_report)


  #Plotting confusion matrix
  sns.heatmap(conf_matrix, annot=True, fmt="d")
  plt.title(f"{model_name} Confusion Matrix")
  plt.ylabel("Actual Values")
  plt.xlabel("Predicted Values")
  plt.show()


  #Plotting ROC-AUC curve
  precisions, recalls, threshold = precision_recall_curve(y_test, y_pred_prob)
  plt.plot(recalls, precisions)
  plt.xlabel("Recall")
  plt.ylabel("Precision")
  plt.title(f"{model_name} Precision-Recall Curve")
  plt.show()


#Put baseline into evaluate
x=evaluate_model("Baseline Logistic Model", baseline_model ,X_test, y_test)
print(x)



#Random Forest model

rf_model = Pipeline([
    ('preprocessor', preprocessor_rf),
    ('classifier', RandomForestClassifier(
        class_weight='balanced',random_state=42))
])
rf_model.fit(X_train, y_train)
y=evaluate_model("rf-model", rf_model ,X_test, y_test)
print(y)




#Hyper parameter tuning of RandomForest
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    "classifier__n_estimators": randint(150, 300),
    "classifier__max_depth": randint(3, 12),
    "classifier__max_samples": uniform(0.6, 0.4),      # rows sampling (0.6-1.0)
    "classifier__max_features": uniform(0.3, 0.6),     # column sampling (0.3-0.9)
    "classifier__min_samples_leaf": randint(1, 10),
    "classifier__min_samples_split": randint(2, 20),
}

random_search = RandomizedSearchCV(
    rf_model,
    param_grid,
    n_iter=50,
    cv=cv,
    scoring="average_precision",
    n_jobs=-1,
    verbose=1,
    random_state=42
)
random_search.fit(X_train, y_train)

#Result after tuning
v=evaluate_model("rf_model", random_search, X_test, y_test)
print(v)


#optimizing the classification threshold
best_model=best_model = random_search.best_estimator_
prob = best_model.predict_proba(X_test)[:, 1]

#For each threshold calculate Precision and Recall
precisions, recalls , thresholds = precision_recall_curve(y_test, prob)


#Calculate F1 score for each threshold
f1 = 2 * (precisions*recalls) / (precisions + recalls + 1e-9)

#Get the threshold that produces the highest f1-score
best = np.argmax(f1[:-1])
best_threshold = thresholds[best]

#Evaluate the Random Forest model using that threshold.
z=evaluate_model("rf_model", best_model, X_test, y_test, best_threshold)
print(z)


#1. Calibrated RandomForest model
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
calibrated_model = CalibratedClassifierCV(
    best_model,
    method="sigmoid",
    cv=5
)

calibrated_model.fit(X_train, y_train)

uncalibrated = rf_model.predict_proba(X_test)[:,1]
calibrated = calibrated_model.predict_proba(X_test)[:,1]
#3. create calibration curves
actual_uncal, predicted_uncal = calibration_curve(y_test, uncalibrated, n_bins=10)
actual_cal, predicted_cal  = calibration_curve(y_test, calibrated, n_bins=10)
#4. plot
plt.plot([0,1], [0,1], 'k--', label="Prefect")
plt.plot(
    predicted_uncal,
    actual_uncal,
    "o-",
    label = "Uncalibrated"
)

plt.plot(
    predicted_cal,
    actual_cal,
    "o-",
    label = "Calibrated"
)
plt.legend()
plt.show()

#SHAP Interpretation — Global & Local
import shap
rf_classifier = best_model.named_steps['classifier']
X_test_transformed = best_model.named_steps['preprocessor'].transform(X_test)
feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)
explainer = shap.TreeExplainer(rf_classifier)
shap_values = explainer.shap_values(X_test_df)

#Global explanation
shap.summary_plot(shap_values, X_test_df)
plt.show()

#Local explanation
row_index = 5
# Class 0
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[row_index, :, 0],
        base_values=explainer.expected_value[0],
        data=X_test_df.iloc[row_index],
        feature_names=feature_names
    )
)

plt.show()

# Class 1
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[row_index, :, 1],
        base_values=explainer.expected_value[1],
        data=X_test_df.iloc[row_index],
        feature_names=feature_names
    )
)

plt.show()


#Analyzing False Positives & False Negatives

y_pred = random_search.predict(X_test)
result_df = X_test.copy()
result_df['actual'] = y_test
result_df['predicted'] = y_pred
false_positive = result_df[(result_df['actual']== 0) & (result_df['predicted']== 1)]
false_negative = result_df[(result_df['actual']== 1) & (result_df['predicted']== 0)]
print(f"False Positive : {len(false_positive)}")
print(f"False Negative : {len(false_negative)}")

#Save the Final Model
import joblib

joblib.dump(calibrated_model, "hr_attrition_model.pkl")

joblib.dump(best_threshold, "best_threshold.pkl")




