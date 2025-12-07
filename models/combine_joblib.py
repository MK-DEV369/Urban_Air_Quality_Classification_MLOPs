import joblib
import numpy as np
from sklearn.ensemble import VotingRegressor

def combine_joblib(chunks, output_file='rf_reg.joblib'):
    estimators = []

    for chunk_file in chunks:
        try:
            model = joblib.load(chunk_file)
            estimators.append((f'model_{len(estimators)}', model))
            print(f"Loaded model from {chunk_file}: {type(model).__name__}")
        except Exception as e:
            print(f"Error loading chunk {chunk_file}: {e}")

    if not estimators:
        print("No models loaded. Cannot create ensemble.")
        return None

    ensemble = VotingRegressor(estimators=estimators)
    joblib.dump(ensemble, output_file)
    print(f"Ensemble model saved to {output_file} with {len(estimators)} base models")

    return ensemble

chunks = ['chunk_0.joblib', 'chunk_1.joblib', 'chunk_2.joblib', 'chunk_3.joblib',
         'chunk_4.joblib', 'chunk_5.joblib', 'chunk_6.joblib', 'chunk_7.joblib',
         'chunk_8.joblib', 'chunk_9.joblib']

combined_model = combine_joblib(chunks)

if combined_model is not None:
    print(f"Combined model type: {type(combined_model).__name__}")
    print(f"Ensemble has {len(combined_model.estimators)} base models")
else:
    print("No ensemble created.")
