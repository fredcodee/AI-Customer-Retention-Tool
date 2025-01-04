import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def load_data(file_path):
    data = pd.read_csv(file_path)

    # Drop non-predictive columns
    columns_to_drop = ['user_id', 'name', 'email', 'canceled_subscription', 'upgrade_date', 'ticket_resolved']
    data_cleaned = data.drop(columns=columns_to_drop)

    # Convert date columns to datetime
    date_columns = ['signup_date', 'last_login_date']
    for col in date_columns:
        data_cleaned[col] = pd.to_datetime(data_cleaned[col])

    # Feature Engineering: Days since signup and last login
    data_cleaned['days_since_signup'] = (pd.Timestamp.now() - data_cleaned['signup_date']).dt.days
    data_cleaned['days_since_last_login'] = (pd.Timestamp.now() - data_cleaned['last_login_date']).dt.days
    data_cleaned = data_cleaned.drop(columns=date_columns)  # Remove raw date columns
    
     # Encode categorical columns
    categorical_columns = ['geographic_location', 'subscription_tier']
    label_encoders = {col: LabelEncoder() for col in categorical_columns}
    for col in categorical_columns:
        data_cleaned[col] = label_encoders[col].fit_transform(data_cleaned[col])

    # Handle missing values (impute with median for numerical, mode for categorical)
    imputer = SimpleImputer(strategy='median')
    data_cleaned.iloc[:, :] = imputer.fit_transform(data_cleaned)

    return data_cleaned
