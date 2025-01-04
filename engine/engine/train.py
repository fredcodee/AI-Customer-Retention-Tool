from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
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

# Hyperparameter Tuning for Logistic Regression and Random Forest
def optimize_hyperparameters(X_train, y_train):
    results = {}

    # Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=500)
    lr_param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    }
    lr_grid = GridSearchCV(lr_model, lr_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    results['Logistic Regression'] = lr_grid.best_estimator_

    # Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    rf_grid = GridSearchCV(rf_model, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    results['Random Forest'] = rf_grid.best_estimator_
    

    return results

# Evaluate Models
def evaluate_models(models, X_test, y_test):
    evaluation_results = {}
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        evaluation_results[model_name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    return evaluation_results