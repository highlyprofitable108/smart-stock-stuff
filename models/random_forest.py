import os
import pandas as pd
import numpy as np
import joblib
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

load_dotenv()


def load_data():
    """Load data from MongoDB into a DataFrame."""
    MONGO_URI = os.getenv('MONGO_URI')
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client.stock_data
    data = pd.DataFrame(list(db.model_data.find()))
    data['date'] = pd.to_datetime(data['date'])
    return data


def feature_engineering(data):
    """Perform feature engineering on the dataset."""
    data['log_volume'] = np.log(data['volume'] + 1)
    data['normalized_ATR'] = (data['ATR'] - data['ATR'].mean()) / data['ATR'].std()
    data['normalized_OBV'] = (data['OBV'] - data['OBV'].mean()) / data['OBV'].std()
    return data


def prepare_data(data):
    """Prepare the dataset for modeling."""
    IGNORE_LIST = [
        '_id', 'company_name', 'date', 'symbol', 'open', 'high', 'low',
        'Bollinger Bands Lower', 'Bollinger Bands Upper',
        'OBV', 'Stochastic', 'MFI', 'Federal Funds Effective Rate',
        'ATR'  # Low importance derived feature
    ]
    data.drop(columns=IGNORE_LIST, inplace=True, errors='ignore')
    features = data.drop('close', axis=1)
    target = data['close']
    return train_test_split(features, target, test_size=0.2, random_state=42), features.columns


def perform_grid_search(X_train, y_train):
    """Perform grid search to find the best model parameters."""
    param_grid = {
        'n_estimators': [10, 50, 100, 200],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def save_model(model, feature_names, directory='data/random_forest'):
    """Save the trained model along with the feature names."""
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save both the model and the feature names as a dictionary
    model_data = {
        'model': model,
        'feature_names': feature_names
    }
    joblib.dump(model_data, os.path.join(directory, 'stock_forecast_model.joblib'))


def generate_detailed_report(model, X_train, X_test, y_train, y_test, predictions, feature_names, directory='data/random_forest'):
    """Generate a detailed report of the model performance."""
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df.sort_values(by="Importance", ascending=False, inplace=True)

    report = f"""
    # Model Training and Evaluation Report

    ## Model Details
    - Model Type: Random Forest Regressor
    - Number of Estimators: {model.n_estimators}
    - Random State: {model.random_state}

    ## Training Data
    - Number of records: {len(X_train)}
    - Features used: {list(X_train.columns)}

    ## Test Data
    - Number of records: {len(X_test)}

    ## Cross-Validation
    - CV Scores (5-fold): {cv_scores}
    - Average CV Score: {np.mean(cv_scores)}

    ## Performance Metrics
    - Mean Squared Error (MSE): {mse}
    - Root Mean Squared Error (RMSE): {rmse}
    - R-squared (R2): {r2}

    ## Feature Importance
    {importance_df.to_string(index=False)}
    """

    with open(os.path.join(directory, 'model_report.md'), 'w') as f:
        f.write(report)


def save_predictions_to_mongodb(X_test, y_test, predictions, db):
    """Save predictions and actual closing prices to MongoDB."""
    collection_name = "model_predictions"
    # Drop the collection if it exists
    if collection_name in db.list_collection_names():
        db[collection_name].drop()
    # Create documents for each prediction with associated features and actual close
    predictions_data = [{
        "features": X_test.iloc[idx].to_dict(),
        "predicted_close": pred,
        "actual_close": actual
    } for idx, (pred, actual) in enumerate(zip(predictions, y_test))]
    db[collection_name].insert_many(predictions_data)


def main():
    data = load_data()
    data = feature_engineering(data)
    (X_train, X_test, y_train, y_test), feature_names = prepare_data(data)
    best_model = perform_grid_search(X_train, y_train)
    predictions = best_model.predict(X_test)

    # Pass feature_names along with the model to the save_model function
    save_model(best_model, feature_names)

    generate_detailed_report(best_model, X_train, X_test, y_train, y_test, predictions, feature_names)


if __name__ == '__main__':
    main()
