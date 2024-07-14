import yfinance as yf
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from celery import Celery
import os
import joblib

model_file_path = os.environ.get('MODEL_FILE_PATH')

app = Flask(__name__)
CORS(app)

app.config['CELERY_BROKER_URL'] = 'redis://redis:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://redis:6379/1'
app.config['CELERY_TASK_TRACK_STARTED'] = True
app.config['CELERY_IGNORE_RESULT'] = False
celery = Celery(app.name, broker='redis://redis:6379/0',
                backend='redis://redis:6379/1')
celery.conf.update(app.config)


def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

# Function to build and train the LSTM model
# def build_lstm_model(sequence_length, num_features):
#     model = Sequential()
#     model.add(LSTM(50, input_shape=(sequence_length, num_features)))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model


def build_lstm_model(sequence_length, num_features):
    model = Sequential()

    model.add(LSTM(128, return_sequences=True,
              input_shape=(sequence_length, num_features)))
    model.add(Dropout(0.2))

    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(32))
    model.add(Dropout(0.2))

    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


@celery.task(bind=True)
def train_model_background(self, ticker, start_date, end_date):
    try:
        stock_data = fetch_stock_data(ticker, start_date, end_date)
        target_column = 'Close'
        stock_prices = stock_data[target_column].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(stock_prices)

        np.save('/app/models/scaled_prices.npy', scaled_prices)

        scaler_filename = '/app/models/scaler.pkl'
        joblib.dump(scaler, scaler_filename)

        sequence_length = 60
        train_data, test_data = scaled_prices[:int(
            0.8*len(scaled_prices))], scaled_prices[int(0.8*len(scaled_prices)):]

        X_train, y_train = create_sequences(train_data, sequence_length)

        num_features = 1
        model = build_lstm_model(sequence_length, num_features)

        epochs = 500
        batch_size = 100
    

        self.update_state(
            state='PROGRESS',
            meta={'message': f'Training in progress'}
        )

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

        # for epoch in range(epochs):
        #     model.fit(X_train, y_train, epochs=1,
        #               batch_size=batch_size, verbose=1)

            # self.update_state(
            #     state='PROGRESS',
            #     meta={'message': f'Training epoch {epoch + 1}/{epochs}'}
            # )

        model.save('/app/models/model.keras')

        self.update_state(state='SUCCESS', meta={
                          'message': 'Model trained and saved successfully!'})
    except Exception as e:
        self.update_state(state='FAILED', meta={'message': str(e)})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})


@app.route('/train_model', methods=['POST'])
def trigger_train_model():
    req_data = request.get_json()
    ticker = req_data['ticker']
    start_date = req_data['start_date']
    end_date = req_data['end_date']

    task = train_model_background.apply_async(
        args=[ticker, start_date, end_date])
    return jsonify({'task_id': task.id}), 202


@app.route('/task_status/<task_id>')
def task_status(task_id):
    task = train_model_background.AsyncResult(task_id)
    if task.state == 'PENDING':
        return jsonify({'status': 'Waiting in queue...'}), 202
    if task.state == 'PROGRESS':
        return jsonify({'status': 'In progress', 'message': task.info.get('message', '')}), 202
    elif task.state == 'SUCCESS':
        return jsonify({'status': 'Completed', 'message': task.info.get('message', '')}), 200
    elif task.state == 'FAILED':
        return jsonify({'status': 'Failed', 'message': task.info.get('message', '')}), 500
    else:
        return jsonify({'status': 'Unhandled'}), 202


@app.route('/forecast_prices', methods=['POST'])
def forecast_prices():
    sequence_length = 60
    req_data = request.get_json()
    forecast_days = int(req_data['forecast_days'])

    loaded_model = load_model('/app/models/model.keras')

    scaled_prices = np.load('/app/models/scaled_prices.npy')
    scaler_filename = '/app/models/scaler.pkl'
    scaler = joblib.load(scaler_filename)

    recent_data = scaled_prices[-sequence_length:]

    forecasts = []
    for _ in range(forecast_days):
        input_data = recent_data.reshape(1, -1, 1)
        predicted_scaled_price = loaded_model.predict(input_data)

        predicted_price = scaler.inverse_transform(predicted_scaled_price)
        forecasts.append(predicted_price[0][0])
        recent_data = np.concatenate((recent_data[1:], predicted_scaled_price))

    forecasts = [float(value) for value in forecasts]
    return jsonify({'predicted_prices': forecasts})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
