# The data used was obtained from Statistics Botswana;Botswana Information and Communication Technology Stats Brief Q1 2023
# Predict future telephone traffic based on historical data ,the user can select the year to predict from a dropdown with options up to 2025
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample data: Domestic and international telephone traffic (minutes) by quarter and year, Q1 2018 - Q1 2023
data = {
    'Year': [2018, 2018, 2018, 2018, 2019, 2019, 2019, 2019, 2020, 2020, 2020, 2020, 2021, 2021, 2021, 2021, 2022, 2022, 2022, 2022, 2023],
    'Quarter': ['Q1', 'Q2', 'Q3', 'Q4', 'Q1', 'Q2', 'Q3', 'Q4', 'Q1', 'Q2', 'Q3', 'Q4', 'Q1', 'Q2', 'Q3', 'Q4', 'Q1', 'Q2', 'Q3', 'Q4', 'Q1'],
    'Domestic_Calls Fixed_to_Fixed': [29051482, 40118866, 29051482, 27180726, 27181299, 23815395, 29851168, 22794450, 24250651, 16386507, 20782825, 20106022, 18627470, 17304785, 15498011, 15776249, 15719621, 15042392, 15381438, 14147411, 13928311],
    'Domestic_Calls Fixed_to_Mobile': [31167605, 31847458, 31167605, 32056516, 31573772, 28819316, 33974218, 26496430, 30205674, 24378849, 27174997, 29399328, 27203594, 25735141, 24958082, 25739964, 26379313, 23970257, 23878393, 23735756, 24129043],
    'Domestic_Calls OnNet_Mobile': [583152907, 248849839, 730436647, 711914725, 1143172248, 1285607439, 1011056590, 985418896, 1100308384, 1644389257, 1777231669, 1787293432, 1818879193, 1860078597, 2074516636, 1998217589, 1903230884, 1945856890, 2087604482, 2071646931, 2010152395],
    'Domestic_Calls OffNet_Mobile': [135349759, 38528098, 165312108, 132472504, 135349759, 142798477, 171590589, 137503751, 187889689, 170571174, 194247474, 208023056, 189090907, 185945710, 178181951, 182451428, 155616501, 148711103, 146054976, 148445942, 130802006],
    'Domestic_Calls Mobile_to_Fixed': [9727092, 3217349, 6737448, 6891007, 9727092, 8329621, 9422874, 9763511, 62961837, 47483422, 59059963, 60450409, 55766233, 52801580, 50780236, 52371979, 58391682, 54840987, 54828695, 54247427, 54079791],
    'International_Calls OutGoing_Fixed': [3411214, 4520062, 3411214, 3434211, 2825652, 2704286, 3744817, 2653915, 2699562, 3234515, 1799422, 1724674, 1482397, 1405280, 1300567, 1222445, 1287234, 1177963, 1119948, 1028440, 949617],
    'International_Calls InComing_Fixed': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'International_Calls OutGoingng_Mobile': [10579435, 3885730, 13482935, 6891130, 10037404, 12628861, 9015650, 8858345, 8216079, 6748608, 6865472, 6858785, 6200439, 5980189, 5818050, 5279882, 4285574, 4867801, 408801, 4119487, 3751063],
    'SMS On_Net': [121579435, 84284330, 174420541, 172698636, 173606036, 132886661, 144579482, 143152172, 130492293, 106028158, 109272506, 110583362, 103449026, 97026663, 100331575, 105694654, 99881824, 98172383, 98580462, 97130920, 87383790],
    'SMS Off_Net': [151872112, 82126432, 177695619, 137566702, 131228779, 144579482, 91645479, 157855498, 80274410, 63583841, 66012639, 68478993, 61083339, 56012965, 55579176, 57968568, 53381690, 51127179, 49534661, 49281498, 45050150],
}

# Create a DataFrame
df = pd.DataFrame(data)

# Extract relevant data
years_quarters = [f"{year} {quarter}" for year, quarter in zip(data['Year'], data['Quarter'])]

# Function to check if all values in a column are zeros
def has_zeros(column):
    return all(value == 0 for value in column)

# Filter out columns with all zeros
filtered_data = {key: value for key, value in data.items() if not has_zeros(value)}

# Bar width
bar_width = 0.15
index = np.arange(len(years_quarters))

# Plotting
st.set_option('deprecation.showPyplotGlobalUse', False)
plt.figure(figsize=(16, 20))

# Plot Domestic Calls
plt.subplot(3, 2, 1)
plt.bar(index - 2*bar_width, data['Domestic_Calls Fixed_to_Fixed'], width=bar_width, label='Fixed to Fixed')
plt.bar(index - bar_width, data['Domestic_Calls Fixed_to_Mobile'], width=bar_width, label='Fixed to Mobile')
plt.bar(index, data['Domestic_Calls OnNet_Mobile'], width=bar_width, label='OnNet Mobile')
plt.bar(index + bar_width, data['Domestic_Calls OffNet_Mobile'], width=bar_width, label='OffNet Mobile')
plt.bar(index + 2*bar_width, data['Domestic_Calls Mobile_to_Fixed'], width=bar_width, label='Mobile to Fixed')
plt.title('Domestic Calls Over Time')
plt.xlabel('Year and Quarter')
plt.ylabel('Number of Calls')
plt.xticks(index, years_quarters, rotation=45, ha='right')
plt.legend()

# Plot International Calls
plt.subplot(3, 2, 2)
plt.bar(index - bar_width, data['International_Calls OutGoing_Fixed'], width=bar_width, label='Outgoing Fixed')
plt.bar(index, data['International_Calls InComing_Fixed'], width=bar_width, label='Incoming Fixed')
plt.bar(index + bar_width, data['International_Calls OutGoingng_Mobile'], width=bar_width, label='Outgoing Mobile')
plt.title('International Calls Over Time')
plt.xlabel('Year and Quarter')
plt.ylabel('Number of Calls')
plt.xticks(index, years_quarters, rotation=45, ha='right')
plt.legend()

# Plot SMS
plt.subplot(3, 2, 3)
plt.bar(index - bar_width, data['SMS On_Net'], width=bar_width, label='On Net')
plt.bar(index + bar_width, data['SMS Off_Net'], width=bar_width, label='Off Net')
plt.title('SMS Over Time')
plt.xlabel('Year and Quarter')
plt.ylabel('Number of SMS')
plt.xticks(index, years_quarters, rotation=45, ha='right')
plt.legend()

# Adjust layout
plt.tight_layout()

# Show plots
st.pyplot()

# Bar width
bar_width = 0.35
index = np.arange(len(years_quarters))

# Plotting
plt.figure(figsize=(12, 6))

# Plot Domestic Calls OnNet vs OffNet Mobile
plt.bar(index - bar_width/2, data['Domestic_Calls OnNet_Mobile'], width=bar_width, label='OnNet Mobile')
plt.bar(index + bar_width/2, data['Domestic_Calls OffNet_Mobile'], width=bar_width, label='OffNet Mobile')
plt.title('Domestic Calls OnNet vs OffNet Mobile Over Time')
plt.xlabel('Year and Quarter')
plt.ylabel('Number of Calls')
plt.xticks(index, years_quarters, rotation=45, ha='right')
plt.legend()

# Show plot
plt.tight_layout()
st.pyplot()

# Choose features and target
features = df[['Year', 'Quarter']]  # You might want to include more features
target = df['Domestic_Calls OnNet_Mobile']  # Choose the target variable

# Convert categorical features to numerical using one-hot encoding
features = pd.get_dummies(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Sidebar for user input
st.sidebar.title("Enter Parameters")

# Dropdown for selecting the year to predict
year_to_predict = st.sidebar.selectbox("Select Year to Predict", list(range(2018, 2026)))

# Predict for a specific quarter and year
quarter_to_predict = st.sidebar.selectbox("Select Quarter to Predict", ['Q1', 'Q2', 'Q3', 'Q4'])

# Use the same one-hot encoding for prediction data
features_to_predict = pd.DataFrame({'Year': [year_to_predict], 'Quarter': [quarter_to_predict]})
features_to_predict = pd.get_dummies(features_to_predict)

# Ensure all columns in the training data are also present in the prediction data
missing_columns = set(features.columns) - set(features_to_predict.columns)
for col in missing_columns:
    features_to_predict[col] = 0

# Reorder the columns to match the order during training
features_to_predict = features_to_predict[features.columns]

# Make the prediction
predicted_traffic = model.predict(features_to_predict)

# Print the predicted value
st.write(f'Predicted Domestic_Calls OnNet_Mobile for {quarter_to_predict} {year_to_predict}: {predicted_traffic[0]}')

# Get the actual value for quarter 1, 2023
actual_value = df.loc[(df['Year'] == year_to_predict) & (df['Quarter'] == quarter_to_predict), 'Domestic_Calls OnNet_Mobile']

if not actual_value.empty:
    actual_value = actual_value.values[0]
    st.write(f'Actual Domestic_Calls OnNet_Mobile for {quarter_to_predict} {year_to_predict}: {actual_value}')
    st.write(f'Difference between Actual and Predicted: {actual_value - predicted_traffic[0]}')
else:
    st.write(f'No actual data available for {quarter_to_predict} {year_to_predict}.')

