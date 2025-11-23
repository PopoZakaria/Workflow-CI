import joblib # type: ignore
import mlflow
import numpy as np 
from sklearn.ensemble import RandomForestRegressor   # type: ignore
from sklearn.metrics import mean_squared_error, r2_score # type: ignore

# Load dataset
x_train = np.load(r'D:\Dicoding\Submission MSML\MLProject\moviesRatingDataset_Processed\x_train.npy')
y_train = np.load(r'D:\Dicoding\Submission MSML\MLProject\moviesRatingDataset_Processed\y_train.npy')
x_val = np.load(r'D:\Dicoding\Submission MSML\MLProject\moviesRatingDataset_Processed\x_val.npy')
y_val = np.load(r'D:\Dicoding\Submission MSML\MLProject\moviesRatingDataset_Processed\y_val.npy')

# training and logging with MLflow
with mlflow.start_run(run_name="RandomForestRegressor") as run:
    # Define model
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    # Fit model
    rf.fit(x_train, y_train)

    # Predict
    y_pred = rf.predict(x_val)

    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    # Log parameters, metrics, and model
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(rf, "random_forest_model")

    # Save model and log model artifact
    joblib.dump(rf, 'random_forest_model.pkl')
    mlflow.log_artifact('random_forest_model.pkl')

    print(f"Model logged in run {run.info.run_id} with RMSE: {rmse} and R2: {r2}")