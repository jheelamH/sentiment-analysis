import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Load the data
df = pd.read_csv("/content/data.csv")

# Preprocess the data
def diagnosis_value(diagnosis):
    return 1 if diagnosis == 'M' else 0

df['diagnosis'] = df['diagnosis'].apply(diagnosis_value)

# Drop the 'Unnamed: 32' and 'id' columns
df = df.drop(['Unnamed: 32', 'id'], axis=1)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Plot the data
sns.lmplot(x='radius_mean', y='texture_mean', hue='diagnosis', data=df_filled)
sns.lmplot(x='smoothness_mean', y='compactness_mean', data=df_filled, hue='diagnosis')

# Plot the distribution of 'area_mean' by diagnosis
sns.displot(data=df_filled, x='area_mean', hue='diagnosis', kind='kde')
plt.title('Distribution of area_mean by Diagnosis')
plt.show()

# Pair plot
sns.pairplot(df_filled, hue='diagnosis')
plt.show()

# Prepare the data for training
X = np.array(df_filled.iloc[:, 1:])
y = np.array(df_filled['diagnosis'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Create pipelines for different classifiers
svm_pipeline = make_pipeline(SimpleImputer(strategy='mean'), SVC())
nb_pipeline = make_pipeline(SimpleImputer(strategy='mean'), GaussianNB())
rf_pipeline = make_pipeline(SimpleImputer(strategy='mean'), RandomForestClassifier())

# Fit and evaluate the models
for name, pipeline in [("SVM", svm_pipeline), ("Naive Bayes", nb_pipeline), ("Random Forest", rf_pipeline)]:
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(f"{name} Score:", score)

    # Perform cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=10, scoring='accuracy')
    print(f"{name} Cross-Validation Mean Score:", np.mean(cv_scores))

# Plot misclassification error for Random Forest
neighbors = []
cv_scores = []
for k in range(1, 51, 2):
    neighbors.append(k)
    rf = RandomForestClassifier(n_estimators=k)
    scores = cross_val_score(rf, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

optimal_n_estimators = neighbors[np.argmin(cv_scores)]
print('The optimal number of estimators for Random Forest is', optimal_n_estimators)

# Plot misclassification error versus k for Random Forest
plt.figure(figsize=(10, 6))
plt.plot(neighbors, cv_scores)
plt.xlabel('Number of estimators')
plt.ylabel('Accuracy')
plt.title('Random Forest: Number of Estimators vs. Accuracy')
plt.show()
