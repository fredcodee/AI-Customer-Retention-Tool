from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Step 3: Train and Evaluate Models
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    results = {}

    # Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=500)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    results['Logistic Regression'] = {
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'classification_report': classification_report(y_test, y_pred_lr),
        'confusion_matrix': confusion_matrix(y_test, y_pred_lr)
    }

    # Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    results['Random Forest'] = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'classification_report': classification_report(y_test, y_pred_rf),
        'confusion_matrix': confusion_matrix(y_test, y_pred_rf)
    }

    return results