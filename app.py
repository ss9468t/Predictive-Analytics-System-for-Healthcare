import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
df = pd.read_csv('Dataset.csv')

# Preprocessing
# Handle missing values
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=[np.number])))
df_imputed.columns = df.select_dtypes(include=[np.number]).columns
df_imputed.index = df.index

# Merge with non-numeric data
df_non_numeric = df.select_dtypes(exclude=[np.number])
df = pd.concat([df_imputed, df_non_numeric], axis=1)

# Convert categorical variables to dummy variables
df = pd.get_dummies(df, drop_first=True)
df['hospital_death'] = df['hospital_death'].apply(lambda x: 1 if x == 1 else 0)
# Split the dataset into features and target variable
X = df.drop('hospital_death', axis=1)
y = df['hospital_death']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Deep Learning Model': Sequential([
        Dense(64, input_dim=X_train.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
}

# Train the models
for model_name, model in models.items():
    if model_name == 'Deep Learning Model':
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    else:
        model.fit(X_train, y_train)

# Streamlit App
st.title('Model Evaluation App')

# Model selection
selected_model = st.selectbox('Choose a model', list(models.keys()))

# Input feature variables
feature_inputs = {}
for feature in X.columns:
    feature_inputs[feature] = st.number_input(f'Enter {feature}', value=0.0)

# Create a DataFrame for prediction
input_data = pd.DataFrame([feature_inputs])
scaled_input_data = scaler.transform(input_data)

# Display classification report and confusion matrix
if st.button('Evaluate Model'):
    selected_model_instance = models[selected_model]
    if selected_model == 'Deep Learning Model':
        y_pred = (selected_model_instance.predict(scaled_input_data) > 0.5).astype('int32')
    else:
        y_pred = selected_model_instance.predict(scaled_input_data)

    st.subheader('Classification Report:')
    st.text(classification_report(y_test, y_pred))

    st.subheader('Confusion Matrix:')
    st.text(confusion_matrix(y_test, y_pred))

    st.subheader('Actual and Predicted Values:')
    st.write(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))
