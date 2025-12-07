import joblib

def split_joblib(file_path, num_chunks=10):
    """
    Create multiple copies of the same model.
    This is useful for distributed inference or ensemble creation.
    """
    model = joblib.load(file_path)

    print(f"Loaded model: {type(model).__name__}")
    print(f"Creating {num_chunks} copies...")

    for i in range(num_chunks):
        chunk_file = f"chunk_{i}.joblib"
        joblib.dump(model, chunk_file)
        print(f"Saved copy {i+1}/{num_chunks}: {chunk_file}")

    return [f"chunk_{i}.joblib" for i in range(num_chunks)]

# Create multiple copies of the Random Forest model
large_model_file = r"E:\5th SEM Data\AI254TA-Machine Learning Operations(MLOps)\MLOPs_Project\models\rf_reg.joblib"
chunks = split_joblib(large_model_file)
print(f"Created {len(chunks)} model copies")
