import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('concrete_data.csv')

# Data cleaning and filtering
df = df[(df['Blast Furnace Slag'] < 330) & (df['Superplasticizer'] < 22) &
        (df['Fine Aggregate'] < 920) & (df['Fine Aggregate'] > 640) &
        (df['Age'] < 80) & (df['Water'] < 230) & (df['Water'] > 130)]

# Split data into features (X) and target (y)
y = df.pop('Strength')
X = df

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)

# List of column names
colm = X_train.columns.tolist()

# Standardize features
scaler = StandardScaler()
X_train[colm] = scaler.fit_transform(X_train[colm])
X_test[colm] = scaler.transform(X_test[colm])

# Linear regression using StatsModels
X_train_lm = sm.add_constant(X_train)
lm1 = sm.OLS(y_train, X_train_lm).fit()

# Variance Inflation Factor (VIF) calculation
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif = vif.sort_values(by="VIF", ascending=False)

# Refine model by removing 'Blast Furnace Slag'
X_train_2 = X_train.drop(["Blast Furnace Slag"], axis=1)
X_train_lm = sm.add_constant(X_train_2)
lm2 = sm.OLS(y_train, X_train_lm).fit()

# VIF calculation after refining model
vif2 = pd.DataFrame()
vif2['Features'] = X_train_2.columns
vif2['VIF'] = [variance_inflation_factor(X_train_2.values, i) for i in range(X_train_2.shape[1])]
vif2 = vif2.sort_values(by="VIF", ascending=False)

# Prediction and Residual Analysis
y_train_pred = lm2.predict(X_train_lm)
res = y_train - y_train_pred

# Create plots
plt.figure(figsize=(10, 6))
sns.distplot(res)
plt.title('Residual Distribution')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_train_pred, res)
plt.axhline(y=0, color='r', linestyle=':')
plt.xlabel("Predictions")
plt.ylabel("Residual")
plt.title('Residual vs Predictions')
plt.show()

plt.figure(figsize=(10, 6))
sm.qqplot(res, fit=True, line='45')
plt.title('QQ Plot of Residuals')
plt.show()

# Prediction on test set and evaluation
X_test_new = X_test[X_train_2.columns]
X_test_lm = sm.add_constant(X_test_new)
y_test_pred = lm2.predict(X_test_lm)
res_test = y_test - y_test_pred

# Print R-squared scores
print("Train R-squared:", r2_score(y_true=y_train, y_pred=y_train_pred))
print("Test R-squared:", r2_score(y_true=y_test, y_pred=y_test_pred))

# Regression plot
plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=y_test_pred, ci=68, fit_reg=True, scatter_kws={"color": "blue"},
            line_kws={"color": "red"})
plt.title('y_test vs y_test_pred')
plt.xlabel('y_test')
plt.ylabel('y_test_pred')
plt.show()

print('Complete')
