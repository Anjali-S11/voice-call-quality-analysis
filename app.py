
import streamlit as st
import pickle
import numpy as np 
import pandas as pd


## load the instances that were created
with open('xgbc_model.pkl','rb') as file:
    final_model = pickle.load(file)
    
# Encoding mappings
operator_mapping = {'Airtel': 0, 'BSNL': 1, 'Idea': 2, 'MTNL': 3, 'Others': 4, 'RJio': 5, 'Tata': 6, 'Vodafone': 7}
travel_mapping = {'Indoor': 0, 'Outdoor': 1, 'Travelling': 2}
network_mapping = {'2G': 0, '3G': 1, '4G': 2, 'Unknown': 3}
call_drop_mapping = {'Call Dropped': 0, 'Poor Voice Quality': 1, 'Satisfactory': 2}
state_mapping = {'Andaman and Nicobar Islands': 0, 'Andhra Pradesh': 1, 'Assam': 2, 'Bihar': 3, 'Chandigarh': 4,
                 'Chhattisgarh': 5, 'Dadra and Nagar Haveli': 6, 'Delhi': 7, 'Goa': 8, 'Gujarat': 9, 'Haryana': 10,
                 'Himachal Pradesh': 11, 'Jharkhand': 12, 'Karnataka': 13, 'Kashmir': 14, 'Kerala': 15, 'Madhya Pradesh': 16,
                 'Maharashtra': 17, 'Meghalaya': 18, 'Odisha': 19, 'Pondicherry': 20, 'Punjab': 21, 'Rajasthan': 22, 'Tamil Nadu': 23,
                 'Telangana': 24, 'Tripura': 25, 'Uttar Pradesh': 26, 'Uttarakhand': 27, 'West Bengal': 28}
month_mapping = {'August': 0, 'July': 1, 'September': 2}
region_mapping = {'Central': 1, 'East': 0, 'North': 2, 'Northeast': 3, 'South': 4, 'West': 5}


# Prediction function
def prediction(Operator, In_Out_Travelling, Network_Type, Call_Drop_Category, Latitude, Longitude, State_Name, Month, Region):
    input_data = np.array([[Operator, In_Out_Travelling, Network_Type, Call_Drop_Category, Latitude, Longitude, State_Name, Month, Region]])
    pred = final_model.predict(input_data)
    return f'Predicted Rating: {pred[0] + 1}'


# Streamlit App
def main():
    # Add an image at the top
    st.image('Picture1.png', width=750)
    st.title('Voice Call Quality Rating Prediction')
    st.write("This application predicts the Voice Call Quality Rating based on network operator, call environment, location, and other factors.")

    # Dropdown menus for categorical inputs (with encoding)
    operator = st.selectbox('Select Operator', ['Select'] + list(operator_mapping.keys()))
    in_out_travel = st.selectbox('In/Out/Travel', ['Select'] + list(travel_mapping.keys()))
    network_type = st.selectbox('Select Network Type', ['Select'] + list(network_mapping.keys()))
    call_drop_category = st.selectbox('Please select the category that best describes your last call experience', ['Select'] + list(call_drop_mapping.keys()))
    state_name = st.selectbox('Select State', ['Select'] + list(state_mapping.keys()))
    month = st.selectbox('Select Month', ['Select'] + list(month_mapping.keys()))
    region = st.selectbox('Select Region', ['Select'] + list(region_mapping.keys()))


    # Numeric inputs for latitude and longitude
    latitude = st.number_input('Enter the latitude of your location in India (Range: 8.4 - 37.6)', min_value=8.4, max_value=37.6, format="%.6f")
    longitude = st.number_input('Enter the longitude of your location in India (Range: 68.7 - 97.25)', min_value=68.7, max_value=97.25, format="%.6f")

    if st.button('Predict'):
        # Convert selections to encoded values
        encoded_operator = operator_mapping[operator]
        encoded_travel = travel_mapping[in_out_travel]
        encoded_network = network_mapping[network_type]
        encoded_call_drop = call_drop_mapping[call_drop_category]
        encoded_state = state_mapping[state_name]
        encoded_month = month_mapping[month]
        encoded_region = region_mapping[region]

        # Make prediction
        response = prediction(encoded_operator, encoded_travel, encoded_network, encoded_call_drop, latitude, longitude, encoded_state, encoded_month, encoded_region)
        st.success(response)

if __name__ == '__main__':
    main()
