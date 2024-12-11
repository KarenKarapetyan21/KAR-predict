import pandas as pd
import numpy as np
import pickle
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
with open('gradient_boosting_model.pkl', 'wb') as f:
    pickle.dump(gb_model, f)
