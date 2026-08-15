# The Complete ML Cheat Sheet
*NumPy · Pandas · Matplotlib/Seaborn/Plotly · Scikit-Learn · Data Loading · Metrics · Loss Functions*
*Goal: everything you need to go from "I have a problem" to "I have a trained, evaluated, saved model."*

---

## 0. How To Use This Sheet

Jump to **Section 16 (End-to-End Flowchart)** first if you just want the order of operations. Every other section is a reference you dip into at the matching step.

---

## 1. Getting Data — Kaggle API, OpenML, Built-in Datasets, Files

### Kaggle API
```bash
pip install kaggle
# Get kaggle.json from kaggle.com/settings -> API -> Create New Token
mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```
```bash
kaggle datasets list -s "titanic"
kaggle datasets download -d heptapod/titanic -p ./data --unzip
kaggle competitions download -c titanic -p ./data
```
```python
import opendatasets as od     # simpler alternative, prompts for kaggle creds
od.download('https://www.kaggle.com/datasets/heptapod/titanic')
```

### OpenML (sklearn integration)
```python
from sklearn.datasets import fetch_openml
data = fetch_openml(name='titanic', version=1, as_frame=True)
df = data.frame
```

### Sklearn's built-in toy/real datasets (for practice)
```python
from sklearn.datasets import (load_iris, load_diabetes, load_digits, load_wine,
                               load_breast_cancer, fetch_california_housing,
                               make_classification, make_regression, make_blobs)
data = load_iris(as_frame=True)
df = data.frame
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2)  # synthetic
```

### Seaborn's bundled datasets (small, clean, good for practice)
```python
import seaborn as sns
df = sns.load_dataset('titanic')   # also: 'tips', 'iris', 'diamonds', 'penguins'
```

### Reading local/remote files into a DataFrame
```python
import pandas as pd

df = pd.read_csv('data.csv')
df = pd.read_csv('https://example.com/data.csv')      # direct URL works too
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df = pd.read_json('data.json')
df = pd.read_parquet('data.parquet')
df = pd.read_sql('SELECT * FROM table', con=connection)
df = pd.read_html('https://example.com/table-page')[0]   # scrapes <table> tags

# Useful read_csv args
pd.read_csv('data.csv', usecols=['a','b'], parse_dates=['date_col'],
            na_values=['NA','?','-'], dtype={'id': str}, nrows=1000)

# Writing back out
df.to_csv('out.csv', index=False)
df.to_parquet('out.parquet')
```

---

## 2. NumPy — Core Ops

```python
import numpy as np

arr = np.array([1, 2, 3])
mat = np.array([[1, 2], [3, 4]])

np.zeros((3,3)); np.ones((2,2)); np.arange(0,10,2); np.linspace(0,1,5)

arr.shape; arr.reshape(3,1); arr.flatten()

arr + 2; arr * 3; arr ** 2
np.dot(mat, mat); mat @ mat

arr.sum(); arr.mean(); arr.std(); arr.min(); arr.max()
mat.sum(axis=0); mat.sum(axis=1)     # column-wise / row-wise

arr[1:3]; mat[0, :]; mat[:, 1]
arr[arr > 1]                          # boolean mask filter

np.random.seed(42)
np.random.rand(3,3); np.random.randint(0,10,5)

np.where(arr > 1, 'big', 'small')     # conditional
np.unique(arr, return_counts=True)
np.concatenate([arr, arr]); np.vstack([mat,mat]); np.hstack([mat,mat])
```

---

## 3. Pandas — Loading, Selecting, Wrangling

```python
df.head(); df.tail(); df.info(); df.describe(); df.shape; df.dtypes

df['col']; df[['col1','col2']]
df.loc[0:5, ['col1','col2']]          # label-based
df.iloc[0:5, 0:2]                     # position-based

df[df['col'] > 10]
df[(df['a'] > 1) & (df['b'] == 'x')]
df.query('a > 1 and b == "x"')

df.isna().sum()
df.dropna(); df.fillna(0); df.fillna(df.mean(numeric_only=True))

df['new'] = df['a'] + df['b']
df['col'] = df['col'].astype('category')
df.rename(columns={'old':'new'})
df.drop(columns=['col'])
df.drop_duplicates()

df.groupby('col')['target'].mean()
df.groupby('col').agg({'target':'mean', 'other':'sum'})
df.pivot_table(index='a', columns='b', values='c', aggfunc='mean')

pd.merge(df1, df2, on='key', how='left')   # how: left/right/inner/outer
pd.concat([df1, df2], axis=0)

df['col'].value_counts()
df['col'].unique(); df['col'].nunique()
df.sort_values('col', ascending=False)
df.apply(lambda row: row['a']+row['b'], axis=1)
df['col'].map({'yes':1,'no':0})
```

---

## 4. Visualization — Matplotlib, Seaborn, Plotly

**Rule of thumb:** Matplotlib = full control/customization. Seaborn = fast statistical plots (built on matplotlib). Plotly = interactive, best for dashboards/exploration.

### Matplotlib
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(x, y); plt.scatter(x, y); plt.bar(categories, values); plt.hist(data, bins=30)
plt.xlabel('X'); plt.ylabel('Y'); plt.title('Title'); plt.legend()
plt.subplot(1,2,1); plt.subplot(1,2,2)          # multiple plots
fig, axes = plt.subplots(2, 2, figsize=(10,8))  # grid of plots
plt.show()
```

### Seaborn (statistical plots, works directly on DataFrames)
```python
import seaborn as sns

sns.histplot(df['col'], kde=True)              # distribution
sns.boxplot(x='category', y='numeric_col', data=df)  # outliers / spread
sns.violinplot(x='category', y='numeric_col', data=df)
sns.scatterplot(x='col1', y='col2', hue='target', data=df)
sns.pairplot(df, hue='target')                 # all pairwise relationships
sns.heatmap(df.corr(), annot=True, cmap='Blues')  # correlation matrix
sns.barplot(x='category', y='value', data=df)
sns.countplot(x='category', data=df)           # categorical counts
sns.lmplot(x='col1', y='col2', data=df)        # scatter + regression line
```

### Plotly (interactive)
```python
import plotly.express as px

px.scatter(df, x='col1', y='col2', color='target', hover_data=['id'])
px.histogram(df, x='col', color='target')
px.box(df, x='category', y='numeric_col')
px.line(df, x='date', y='value')
px.bar(df, x='category', y='value')
fig = px.scatter_3d(df, x='a', y='b', z='c', color='target')
fig.show()
```

### What to plot, when (EDA checklist)
| Goal | Plot |
|---|---|
| Distribution of one numeric column | `histplot`, `boxplot` |
| Distribution of one categorical column | `countplot`, `value_counts().plot(kind='bar')` |
| Relationship between 2 numeric columns | `scatterplot`, `lmplot` |
| Relationship between numeric & categorical | `boxplot`, `violinplot`, `barplot` |
| Relationships among ALL numeric columns | `pairplot`, `corr()` + `heatmap` |
| Target imbalance (classification) | `countplot(x=target)` |
| Outlier detection | `boxplot` |
| Time trends | `lineplot` / `px.line` |

---

## 5. EDA — General Pattern (before any model)

1. `df.head()`, `df.info()`, `df.describe()` — shape/types/stats
2. `df.isna().sum()` — missing values
3. Target distribution — `value_counts()` (classification) / `hist()` (regression)
4. Univariate — histograms/boxplots per numeric column
5. Bivariate — correlation with target, scatterplots, groupby comparisons
6. Categorical columns — `value_counts()`, bar plots
7. Outliers — boxplots, IQR method
8. Correlations — `df.corr()`, `sns.heatmap()`
9. Check for duplicate rows — `df.duplicated().sum()`
10. Check class imbalance if classification — `df[target].value_counts(normalize=True)`

---

## 6. Data Cleaning & Transforming Tools

### Missing values
```python
from sklearn.impute import SimpleImputer, KNNImputer

SimpleImputer(strategy='mean')       # numeric: mean/median
SimpleImputer(strategy='most_frequent')  # categorical
KNNImputer(n_neighbors=5)            # smarter, uses similar rows
```

### Outliers
```python
Q1, Q3 = df['col'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df_clean = df[(df['col'] >= Q1 - 1.5*IQR) & (df['col'] <= Q3 + 1.5*IQR)]

from scipy import stats
df[(np.abs(stats.zscore(df[numeric_cols])) < 3).all(axis=1)]  # z-score method
```

### Scaling (numeric features)
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

StandardScaler()   # mean=0, std=1 — use for linear/logistic reg, SVM, KNN, PCA
MinMaxScaler()     # scales to [0,1] — use when you need bounded range
RobustScaler()     # uses median/IQR — use when data has outliers
```

### Encoding categorical data
```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # nominal, no order
OrdinalEncoder()          # ordinal, has order (e.g. Low/Med/High)
LabelEncoder()             # for target column only (not features!)

# pandas shortcut
pd.get_dummies(df, columns=['cat_col'], drop_first=True)
```

### Feature engineering
```python
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year; df['month'] = df['date'].dt.month
df['dayofweek'] = df['date'].dt.dayofweek

df['log_col'] = np.log1p(df['col'])           # reduce skew
df['binned'] = pd.cut(df['col'], bins=5)       # bucket numeric -> categorical
df['interaction'] = df['a'] * df['b']          # interaction feature

from sklearn.preprocessing import PolynomialFeatures
PolynomialFeatures(degree=2)                   # generate polynomial/interaction terms
```

### Handling class imbalance
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)  # oversample minority
# or: use class_weight='balanced' in most sklearn models instead
```

### Text cleaning (basic, before NLP-specific tools like spaCy)
```python
df['text'] = df['text'].str.lower().str.replace(r'[^a-z\s]', '', regex=True)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
X_text = TfidfVectorizer(max_features=5000).fit_transform(df['text'])
```

---

## 7. Splitting Data — Every Method

```python
from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold,
                                      TimeSeriesSplit, GroupKFold)

# Basic split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Stratified split (preserves class ratio — use for imbalanced classification)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['target'], random_state=42)

# 3-way split (train/val/test)
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)

# K-Fold cross-validation (more robust than one split)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # for classification

# Time series — NEVER shuffle, always split chronologically
tscv = TimeSeriesSplit(n_splits=5)

# Grouped data — e.g. multiple rows per patient/user, keep groups together
gkf = GroupKFold(n_splits=5)
```
| Situation | Use |
|---|---|
| Generic tabular data | `train_test_split` |
| Imbalanced classes | `train_test_split(..., stratify=y)` or `StratifiedKFold` |
| Small dataset, need robust estimate | `KFold` / `StratifiedKFold` with `cross_val_score` |
| Time series data | `TimeSeriesSplit` (never random shuffle!) |
| Multiple rows belong to same entity | `GroupKFold` |

---

## 8. Scikit-Learn — Universal Workflow

```
1. Get the data (Section 1)
2. EDA (Sections 4-5)
3. Create Train / Val / Test sets (Section 7)
4. Identify Input and Target columns
5. Identify numeric vs categorical columns
6. Impute missing numeric data (Section 6)
7. Scale numeric features (Section 6)
8. Encode categorical data (Section 6)
9. (Optional) Save processed data to disk
10. Train model -> Predict -> Evaluate -> Save/Load model
```
```python
input_cols = list(df.columns)[1:-1]
target_col = 'target'
train_inputs, train_targets = train_df[input_cols].copy(), train_df[target_col].copy()
val_inputs, val_targets = val_df[input_cols].copy(), val_df[target_col].copy()
test_inputs, test_targets = test_df[input_cols].copy(), test_df[target_col].copy()

numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()

imputer = SimpleImputer(strategy='mean').fit(train_inputs[numeric_cols])
scaler = MinMaxScaler().fit(train_inputs[numeric_cols])
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_inputs[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

for split_inputs in [train_inputs, val_inputs, test_inputs]:
    split_inputs[numeric_cols] = imputer.transform(split_inputs[numeric_cols])
    split_inputs[numeric_cols] = scaler.transform(split_inputs[numeric_cols])
    split_inputs[encoded_cols] = encoder.transform(split_inputs[categorical_cols])

X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]

import joblib
joblib.dump(model, 'model.joblib')
loaded_model = joblib.load('model.joblib')
```

---

## 9. Model Catalog — Every Major Scikit-Learn Model, When To Use It

### Regression (predicting a continuous number)
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
```
| Model | Use When | Notes |
|---|---|---|
| `LinearRegression` | Linear relationship, interpretability matters | Baseline, fast |
| `Ridge` | Linear + many correlated features | L2 regularization, reduces overfitting |
| `Lasso` | Linear + want automatic feature selection | L1 regularization, zeros out weak features |
| `ElasticNet` | Mix of Ridge + Lasso benefits | Good default regularized linear model |
| `DecisionTreeRegressor` | Non-linear, need interpretability | Overfits without `max_depth` |
| `RandomForestRegressor` | Non-linear, general purpose, robust | Strong baseline for tabular data |
| `GradientBoostingRegressor` / `XGBoost` / `LightGBM` | Need best accuracy on tabular data | Slower to tune, best Kaggle performance |
| `SVR` | Small-medium data, non-linear via kernel | Needs scaled features, slow on big data |
| `KNeighborsRegressor` | Simple, local patterns matter | Slow at prediction time, needs scaling |

### Classification (predicting a category/label)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
```
| Model | Use When | Notes |
|---|---|---|
| `LogisticRegression` | Binary/multi-class, need interpretability & probabilities | Fast, strong baseline |
| `DecisionTreeClassifier` | Need interpretable rules | Overfits without depth limit |
| `RandomForestClassifier` | General purpose tabular classification | Robust default choice |
| `GradientBoostingClassifier` / `XGBoost` / `LightGBM` / `CatBoost` | Best accuracy needed | Standard for Kaggle-style tabular problems |
| `SVC` | Small-medium data, clear margin between classes | Needs scaling, slow on large data |
| `KNeighborsClassifier` | Simple baseline, local similarity matters | Needs scaling, slow at inference |
| `GaussianNB` | Text/spam classification, fast baseline | Assumes feature independence |
| `MultinomialNB` | Text classification (word counts/TF-IDF) | Very fast, works well on sparse text data |

### Clustering (no labels)
```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
```
| Model | Use When |
|---|---|
| `KMeans` | Know approx. number of clusters, round/equal-sized groups |
| `DBSCAN` | Don't know cluster count, irregular shapes, want outlier detection |
| `AgglomerativeClustering` | Want a hierarchy/dendrogram of relationships |

### Dimensionality Reduction
```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
```
| Model | Use When |
|---|---|
| `PCA` | Reduce features while keeping variance, speed up training, remove multicollinearity |
| `TSNE` | Visualize high-dim data in 2D/3D (not reusable on new data) |

### Anomaly Detection
```python
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
```

---

## 10. Evaluation Metrics — Full Reference

### Regression
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

mean_squared_error(y_true, y_pred, squared=False)   # RMSE — penalizes large errors more
mean_absolute_error(y_true, y_pred)                  # MAE — average absolute error, robust to outliers
r2_score(y_true, y_pred)                             # R² — variance explained (1.0 = perfect)
mean_absolute_percentage_error(y_true, y_pred)       # MAPE — error as a %, good for business reporting
```
| Metric | Use When |
|---|---|
| RMSE | Standard choice; large errors should be penalized heavily |
| MAE | Outliers present and shouldn't dominate the score |
| R² | Want a normalized "how good is my model" (0-1) |
| MAPE | Need error explained in percentage terms to stakeholders |

### Classification
```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, classification_report,
                              log_loss, precision_recall_curve, roc_curve)

accuracy_score(y_true, y_pred)
precision_score(y_true, y_pred)     # of predicted positives, how many correct?
recall_score(y_true, y_pred)        # of actual positives, how many found?
f1_score(y_true, y_pred)            # harmonic mean of precision & recall
roc_auc_score(y_true, y_probs)      # ranking quality across all thresholds
confusion_matrix(y_true, y_pred)
print(classification_report(y_true, y_pred))
```
| Metric | Use When |
|---|---|
| Accuracy | Classes are balanced |
| Precision | False positives are costly (e.g. spam filter, fraud flag) |
| Recall | False negatives are costly (e.g. disease detection, fraud missed) |
| F1-score | Need balance between precision & recall, classes imbalanced |
| ROC-AUC | Care about ranking/probability quality, not just hard labels |
| Confusion Matrix | Want the full breakdown of TP/FP/TN/FN |
| Log Loss | Model outputs probabilities and you want to penalize confident wrong answers |

### Clustering (no ground truth)
```python
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

silhouette_score(X, labels)         # higher = better separated clusters (-1 to 1)
davies_bouldin_score(X, labels)     # lower = better
calinski_harabasz_score(X, labels)  # higher = better
```

---

## 11. Loss Functions — What Models Actually Minimize During Training

Loss functions are **not the same as evaluation metrics** — loss is what the optimizer minimizes during training (needs to be differentiable); metrics are what you report to judge the model afterward. Sometimes they're the same thing (e.g. MSE), sometimes not (e.g. accuracy isn't differentiable, so classifiers train on log loss instead).

### Regression loss functions
| Loss | Formula idea | Used by |
|---|---|---|
| **MSE** (Mean Squared Error) | avg of (y_true - y_pred)² | `LinearRegression`, neural nets (default) |
| **MAE** (Mean Absolute Error) | avg of \|y_true - y_pred\| | Robust regression, less sensitive to outliers |
| **Huber Loss** | MSE for small errors, MAE for large errors | `HuberRegressor` — best of both worlds |

### Classification loss functions
| Loss | Formula idea | Used by |
|---|---|---|
| **Log Loss / Binary Cross-Entropy** | penalizes confident wrong probability predictions | `LogisticRegression`, neural nets (binary) |
| **Categorical Cross-Entropy** | multi-class version of log loss | Neural nets (multi-class), softmax outputs |
| **Hinge Loss** | maximizes margin between classes | `SVC` (SVM) |
| **Gini Impurity / Entropy** | measures node "purity" for splits | `DecisionTreeClassifier` (`criterion='gini'/'entropy'`) |
| **Exponential Loss** | heavily penalizes misclassified points | `AdaBoost` |

```python
# Setting loss/criterion explicitly where sklearn exposes it
DecisionTreeClassifier(criterion='gini')       # or 'entropy', 'log_loss'
DecisionTreeRegressor(criterion='squared_error')  # or 'absolute_error', 'friedman_mse'
SGDClassifier(loss='log_loss')                 # or 'hinge' (=SVM), 'modified_huber'
HuberRegressor()                               # built-in Huber loss regressor
```

---

## 12. Unsupervised Learning — Clustering (workflow)

No target column — pipeline shrinks: EDA → Scale → Fit → Evaluate (silhouette) → Interpret.
```python
from sklearn.cluster import KMeans
X_scaled = StandardScaler().fit_transform(df[numeric_cols])

inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    inertias.append(km.inertia_)
plt.plot(range(1,11), inertias, marker='o')   # elbow method

model = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = model.fit_predict(X_scaled)
silhouette_score(X_scaled, df['cluster'])
```
```python
from sklearn.cluster import DBSCAN
labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_scaled)  # -1 = outlier
```

---

## 13. Unsupervised Learning — Dimensionality Reduction (workflow)

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)          # or n_components=0.95 to keep 95% variance
X_pca = pca.fit_transform(X_scaled)
pca.explained_variance_ratio_

from sklearn.manifold import TSNE
X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X_scaled)
```
**PCA vs t-SNE:** PCA is linear, fast, reusable via `.transform()` on new data. t-SNE is non-linear, better for visualization, can't transform new points consistently.

---

## 14. Pipelines & Hyperparameter Tuning (production-grade glue)

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

preprocessor = ColumnTransformer([
    ('num', Pipeline([('impute', SimpleImputer(strategy='mean')),
                       ('scale', StandardScaler())]), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
])
pipe = Pipeline([('preprocessor', preprocessor),
                  ('model', RandomForestClassifier(random_state=42))])
pipe.fit(train_inputs, train_targets)      # one call handles everything

scores = cross_val_score(pipe, train_inputs, train_targets, cv=5, scoring='f1')

grid = GridSearchCV(pipe, param_grid={'model__n_estimators': [50,100,200],
                                       'model__max_depth': [5,10,None]}, cv=5, scoring='f1')
grid.fit(train_inputs, train_targets)
grid.best_params_, grid.best_score_
```
**Why:** wraps impute→scale→encode→model into ONE object; `.fit()`/`.predict()` does everything, and there's zero risk of leaking val/test info into training stats.

---

## 15. Reinforcement Learning — Outside Scikit-Learn

Scikit-learn only covers **supervised** (X→y) and **unsupervised** (X only) learning — no RL module. RL is agent + environment + reward, a different paradigm entirely.
```python
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make('CartPole-v1')
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
```
| Library | Purpose |
|---|---|
| `gymnasium` | Standard RL environments |
| `stable-baselines3` | Ready algorithms (PPO, DQN, A2C, SAC) on PyTorch |
| `RLlib` (Ray) | Distributed/scalable RL |

This will most likely resurface later under **AI Agents / RLHF** on your roadmap, not as a scikit-learn topic.

---

## 16. End-to-End Flowchart — "Just Follow This For Any ML Problem"

```
1.  Define the problem
    -> Regression (predict a number)? Classification (predict a category)?
       Clustering (no labels)? Time series?

2.  Get the data
    -> Kaggle API / OpenML / sklearn.datasets / CSV file  (Section 1)

3.  Load into pandas, first look
    -> df.head(), df.info(), df.describe(), df.shape        (Section 3)

4.  EDA
    -> Missing values, target distribution, correlations,
       univariate & bivariate plots                          (Sections 4-5)

5.  Clean the data
    -> Handle duplicates, outliers, wrong dtypes,
       feature engineering (dates, log transforms, bins)      (Section 6)

6.  Split the data
    -> train_test_split / StratifiedKFold / TimeSeriesSplit   (Section 7)

7.  Identify input/target columns, numeric/categorical cols   (Section 8)

8.  Preprocess
    -> Impute missing values -> Scale numeric -> Encode categorical (Section 6)
    -> (Best practice: wrap in a Pipeline, Section 14)

9.  Pick a model family based on the problem type & data size (Section 9)
    -> Start with a simple baseline (Linear/Logistic Regression)
    -> Then try a tree ensemble (Random Forest / XGBoost) for better accuracy

10. Train
    -> model.fit(X_train, y_train)

11. Predict
    -> model.predict(X_val)  /  model.predict_proba(X_val)

12. Evaluate with the RIGHT metric for your problem            (Section 10)
    -> Regression: RMSE / MAE / R²
    -> Classification: Accuracy / Precision / Recall / F1 / ROC-AUC
    -> Imbalanced classes? Don't trust accuracy -> use F1 or ROC-AUC

13. Diagnose
    -> Train score >> Val score? Overfitting -> reduce depth,
       add regularization, get more data, simplify model
    -> Train score ≈ Val score but both low? Underfitting ->
       more complex model, better features, less regularization

14. Tune hyperparameters
    -> GridSearchCV / RandomizedSearchCV with cross-validation  (Section 14)

15. Finalize
    -> Retrain best model on train+val combined
    -> Evaluate ONCE on the held-out test set (never touched before)

16. Save
    -> joblib.dump(model, 'model.joblib')                       (Section 8)

17. Deploy / predict on new data
    -> Load model + preprocessing objects, transform new input
       the same way, then model.predict()
```

---

## 17. Cross-Model Comparison Table

| Model | Needs Scaling? | Handles Categorical Natively? | Overfits Easily? | Interpretability | Typical Use |
|---|---|---|---|---|---|
| Linear/Logistic Regression | Yes | No | Low | High | Linear relationships, interpretable baseline |
| Decision Tree | No | No (sklearn needs encoding) | High | High | Simple rules, quick baseline |
| Random Forest | No | No | Medium | Medium | Robust general-purpose tabular model |
| Gradient Boosting/XGBoost | No | No | Medium (tunable) | Low | Best raw accuracy on tabular data |
| SVM (SVC/SVR) | Yes | No | Medium | Low | Small-medium data, clear margins |
| KNN | Yes | No | Medium | High | Simple baseline, local patterns |
| Naive Bayes | No | Partial | Low | High | Text classification, fast baseline |
| K-Means | Yes | No | N/A (unsupervised) | Medium | Segmentation, grouping |

---

*Only three things really change per model: **which class you import**, **whether it needs scaling**, and **which metric you evaluate with**. Everything else — EDA, splitting, imputing, encoding, saving — is the same pipeline every time.*
