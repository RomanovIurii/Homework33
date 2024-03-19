import os
import pandas as pd
import dill
import json
import glob

def predict():
    def load_model(model_filename):
        with open(model_filename, 'rb') as file:
            model = dill.load(file)
        return model

    def load_json_data(json_filename):
        with open(json_filename, 'r') as file:
            data = json.load(file)
        return data

    def predict_with_model(model, data):
        X = pd.DataFrame([data])
        predictions = model.predict(X)
        return data['id'], predictions

    models_folder = os.path.join(os.environ.get('PROJECT_PATH', '/Users/ninaromanova/airflow_hw'), 'data', 'models')
    model_files = glob.glob(os.path.join(models_folder, '*.pkl'))
    latest_model_file = max(model_files, key=os.path.getctime)

    model = load_model(latest_model_file)

    test_data_folder = os.path.join(os.environ.get('PROJECT_PATH', '/Users/ninaromanova/airflow_hw'), 'data', 'test')
    predictions = []

    for filename in os.listdir(test_data_folder):
        if filename.endswith('.json'):
            json_filepath = os.path.join(test_data_folder, filename)
            data = load_json_data(json_filepath)
            id_, prediction = predict_with_model(model, data)
            predictions.append({'id': id_, 'predicted_price_category': prediction})

    predictions_df = pd.DataFrame(predictions, columns=['id', 'predicted_price_category'])

    predictions_folder = os.path.join(os.environ.get('PROJECT_PATH', '/Users/ninaromanova/airflow_hw'), 'data', 'predictions')
    if not os.path.exists(predictions_folder):
        os.makedirs(predictions_folder)

    predictions_csv_path = os.path.join(predictions_folder, 'predictions.csv')
    predictions_df.to_csv(predictions_csv_path, index=False)

    print(f"Predictions saved to: {predictions_csv_path}")

    predictions_df = pd.read_csv(predictions_csv_path)

    print(predictions_df)

if __name__ == '__main__':
    predict()