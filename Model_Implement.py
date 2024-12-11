import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, make_scorer
from sklearn.ensemble import GradientBoostingRegressor

# Տվյալների ներբեռնում
df = pd.read_csv('NEW_DIGITAL_data.csv')

# Գնի քվարտիլների հաշվարկ և աուտլայերների հեռացում
Q1 = df['Գին'].quantile(0.25)
Q3 = df['Գին'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['Գին'] >= lower_bound) & (df['Գին'] <= upper_bound)]

# Հատկանիշների ընտրություն
selected_features = [
    'Ընդհանուր մակերես', 'Սենյակների քանակ', 'Հարկ', 'Շինության տիպ', 'Վերանորոգում', 'Նորակառույց', 'Համայնք',
    'Գին_մեկ_մետրի_համար', 'Մակերես_մեկ_սենյակի_համար','Հին–Հայտարարություն'
]

X = df[selected_features]
y = df['Գին']

# Տվյալների բաժանում
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Ստեղծում ենք Gradient Boosting մոդելը
gb_model = GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

# Ուսուցանում ենք մոդելը
gb_model.fit(X_train_full, y_train_full)

# Գնի կանխատեսում Gradient Boosting մոդելի միջոցով
y_pred_gb = gb_model.predict(X_test)

# Պահպանում ենք մոդելը
joblib.dump(gb_model, 'gradient_boosting_model.pkl')

# Մոդելի բեռնում
loaded_model = joblib.load('gradient_boosting_model.pkl')

# Կատարողականության գնահատում
mae_gb = mean_absolute_error(y_test, y_pred_gb)
print(f"Gradient Boosting-ի Mean Absolute Error (MAE): {mae_gb}")

r2_gb = r2_score(y_test, y_pred_gb)
print(f"Gradient Boosting-ի R²: {r2_gb}")

# Հաշվում ենք Mean Squared Error (MSE) և Root Mean Squared Error (RMSE)
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)

print(f"Gradient Boosting-ի Mean Squared Error (MSE): {mse_gb}")
print(f"Gradient Boosting-ի Root Mean Squared Error (RMSE): {rmse_gb}")

# Cross-validation գնահատում
def cross_val_metrics(model, X, y, cv):
    scorers = {
        'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
        'MSE': make_scorer(mean_squared_error, greater_is_better=False),
        'R2': make_scorer(r2_score),
        'RMSE': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)
    }

    results = {}
    for metric, scorer in scorers.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
        results[metric] = -np.mean(scores) if metric in ['MAE', 'MSE', 'RMSE'] else np.mean(scores)
    return results

cv_results = cross_val_metrics(gb_model, X_train_full, y_train_full, cv=5)

print("Cross-Validation Results:")
for metric, score in cv_results.items():
    print(f"{metric}: {score}")

# Հատկանիշների կարևորությունների ստացում
feature_importances_gb = gb_model.feature_importances_
importance_df_gb = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances_gb
}).sort_values(by='Importance', ascending=False)

print(importance_df_gb)