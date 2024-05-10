import json
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

with open('data.json', 'r') as f:
    input_data = json.load(f)

with open('out_data.json', 'r') as o:
    output_data = json.load(o)


def prepare_data(input_data, max_text_length=100):
    prepared_data = []
    all_text_data = []

    for project in input_data:
        text_data = [
            project["project_name"],
            project["project_direction"],
            project["project_goal"],
            project["client_name"],
            project["project_summary"]
        ]
        all_text_data.extend(text_data)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_text_data)

    for project in input_data:
        numeric_values = [
            int(project["metric"]),
            len(project["project_name"]),
            len(project["project_direction"]),
            len(project["project_goal"]),
            len(project["client_name"]),
            len(project["project_summary"]),
            project["budget"]["amount"],
            project["estimated_actual_time"],
            len(project["team_composition"]),
            int(project["subject_to_changes"])
        ]

        text_data = [
            project["project_name"],
            project["project_direction"],
            project["project_goal"],
            project["client_name"],
            project["project_summary"]
        ]

        text_sequences = tokenizer.texts_to_sequences(text_data)
        padded_sequences = pad_sequences(text_sequences, maxlen=max_text_length)

        flattened_sequences = padded_sequences.flatten()

        combined_features = np.concatenate([numeric_values, flattened_sequences])

        prepared_data.append(combined_features)

    return np.array(prepared_data)

def format_output(risks):
    risk_keys = list(output_data[0]['risks'].keys())
    formatted_output = {}
    for i, key in enumerate(risk_keys):
        formatted_output[key] = round(float(risks[i]), 2)
    return formatted_output



X = prepare_data(input_data)
y = np.array([list(item['risks'].values()) for item in output_data])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

input_dim = X_scaled.shape[1]

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(1024, input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dense(11, activation='linear'))

optimizer = Adam(learning_rate=0.0001)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])

early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=320, batch_size=10, validation_data=(X_val, y_val), callbacks=[early_stopping])
#model.save('model.keras')

model = keras.saving.load_model("model.keras")

nn_val_predictions = model.predict(X_val)
nn_val_mse = mean_squared_error(y_val, nn_val_predictions)
print(f"Neural Network MSE on validation set: {nn_val_mse}")



rf_model = RandomForestRegressor(n_estimators=5000, random_state=42)
rf_model.fit(X_train, y_train)

rf_val_predictions = rf_model.predict(X_val)
rf_val_mse = mean_squared_error(y_val, rf_val_predictions)
print(f"Random Forest MSE on validation set: {rf_val_mse}")

def predictRisks(data):
    X_new = prepare_data(data)
    X_new_scaled = scaler.transform(X_new)

    nn_predictions = model.predict(X_new_scaled)
    rf_predictions = rf_model.predict(X_new_scaled)

    combined_predictions = (nn_predictions + rf_predictions) / 1

    predicted_risks = [format_output(prediction) for prediction in combined_predictions]
    return json.dumps(predicted_risks, indent=4)




testData = [{
    "id": 2,
    "metric": "3",
    "project_name": "Development of an AI-powered chatbot for customer support",
    "project_direction": "artificial intelligence",
    "project_goal": "Create a sophisticated AI-powered chatbot to enhance customer service operations and provide 24/7 support.",
    "client_name": "Dynamic Support Solutions",
    "project_summary": "This project aims to develop an intelligent chatbot that uses natural language processing to understand and respond to customer inquiries. The chatbot will be integrated into websites and mobile apps to provide instant support, reduce response times, and improve customer satisfaction.",
    "budget": {
        "amount": 6,
        "currency": "USD"
    },
    "project_timeline": {
        "from": "2024-10-01",
        "to": "2025-04-01"
    },
    "estimated_actual_time": 2,
    "team_composition": [
        {
            "full_name": "Ivan Petrov",
            "experience": "8 years",
            "projects_completed": 18,
            "brief_description": "Specializes in artificial intelligence and machine learning, with successful implementations in customer interaction systems.",
            "education": "Higher education, Department of Artificial Intelligence"
        }
    ],
    "subject_to_changes": False
}

]

print(predictRisks(testData))