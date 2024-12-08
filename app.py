import streamlit as st
import pandas as pd
import numpy as np
import pickle


filename1 = 'heart-disease-prediction-age-gradientBoost-modell.pkl'
model4 = pickle.load(open(filename1, 'rb'))

filename2 = 'heart-disease-prediction-bp-modell.pkl'
model2 = pickle.load(open(filename2, 'rb'))

filename3 = 'heart-disease-prediction-Excersice-decisiontree-modell.pkl'
model3 = pickle.load(open(filename3, 'rb'))

filename4 = 'heart-disease-prediction-smoking-modell.pkl'
model1 = pickle.load(open(filename4, 'rb'))

# Function to make predictions
def predict_heart_disease(inputs):
    input_array = np.array(inputs).reshape(1, -1)
    prediction = model1.predict(input_array)
    return prediction[0]

# Save the updated DataFrame to CSV
def save_to_csv():
    st.session_state.df.to_csv('csvcleaned_heart_attack_prediction_dataset.csv', index=False)

# Load the dataset into session state
if 'df' not in st.session_state:
    st.session_state.df = pd.read_csv('csvcleaned_heart_attack_prediction_dataset.csv')

# Title for the web app
st.title('Artery Disease Prediction App')

# Create horizontal tabs
tabs = st.selectbox('Select a Section:', ['Check My Heart Health', 'View All Record', 'Add a Record', 'Update a Record', 'Delete Record'])

# --- Tab 1: Check My Heart Health ---
if tabs == 'Check My Heart Health':
    st.header('Check Your Heart Health')

    # Form inputs for user to fill out
    # Form inputs
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    cholesterol = st.number_input("Cholesterol (in mg/dl)", min_value=100, max_value=600, step=1)
    blood_pressure = st.text_input("Blood Pressure (systolic_bp/diastolic_bp) (e.g., 120/80)")
    heart_rate = st.number_input("Heart Rate", min_value=30, max_value=200, step=1)
    diabetes = st.selectbox("Diabetes", ["Yes", "No"])
    smoking = st.selectbox("Smoking", ["Yes", "No"])
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, step=0.1)
    physical_activity = st.number_input("Days of Physical Activity per Week", min_value=0, max_value=7, step=1)
  
    # All fields are mandatory
    if st.button("Predict"):
        # Validate mandatory fields
        if age == 0 or cholesterol == 0 or heart_rate == 0 or bmi == 0:
            st.error("Please fill in all fields. All fields are mandatory and should not be 0.")
        elif blood_pressure == "" or len(blood_pressure.split("/")) != 2:
            st.error("Please enter blood pressure in the format: systolic/diastolic (e.g., 120/80).")
        else:
            # Extract systolic and diastolic BP from the input
            try:
                systolic_bp, diastolic_bp = map(int, blood_pressure.split('/'))
            except ValueError:
                st.error("Please enter valid numeric values for blood pressure (e.g., 120/80).")
            else:
                    gender = 1 if gender == "Male" else 0
                    diabetes = 1 if diabetes == "Yes" else 0
                    smoking = 1 if smoking == "Yes" else 0

                    # Input list for prediction
                    inputs = [
                        age, gender, cholesterol, systolic_bp, diastolic_bp, heart_rate,
                        diabetes, smoking, bmi, physical_activity
                    ]
                    input1 = [smoking,heart_rate]
                    input2 = [systolic_bp, diastolic_bp]
                    input3 = [physical_activity]
                    input4 = [age]

                    # Make predictions
                    input_array4 = np.array(input4).reshape(1, -1)
                    input_array1 = np.array(input1).reshape(1, -1)
                    input_array2 = np.array(input2).reshape(1, -1)
                    input_array3 = np.array(input3).reshape(1, -1)

                    prediction1 = model1.predict(input_array1)[0]
                    prediction2 = model2.predict(input_array2)[0]
                    prediction3 = model3.predict(input_array3)[0]
                    prediction4 = model4.predict(input_array4)[0]
                    
                    # prediction2 = model2.predict(input_array)[0]
                    # prediction3 = model3.predict(input_array)[0]
                    # prediction4 = model4.predict(input_array)[0]

                    # Calculate mean prediction
                    mean_prediction = np.mean([prediction1, prediction2, prediction3, prediction4])

                    # Display the result
                    # Categorize the risk based on mean_prediction
                    if mean_prediction < 0.25:
                        st.success(f"You are at low risk of having artery disease. (Mean Risk Score: {mean_prediction:.2f})")
                    elif 0.25 <= mean_prediction < 0.5:
                        st.success(f"You are at below average risk of having artery disease. (Mean Risk Score: {mean_prediction:.2f})")
                    elif 0.5 <= mean_prediction < 0.75:
                        st.success(f"You are at above average risk of having artery disease. You need to see a doctor.(Mean Risk Score: {mean_prediction:.2f})")
                    else:
                        st.success(f"You are at high risk of having artery disease. You immediately need to see a doctor (Mean Risk Score: {mean_prediction:.2f})")

# --- Tab 2: View Record ---

elif tabs == 'View All Record':

    st.header('View Heart Disease Records')

    # Show the dataset
    st.dataframe(st.session_state.df)

# --- Tab 3: Add a Record ---
elif tabs == 'Add a Record':
    st.header('Add a New Record')

    new_record = {}
    all_filled = True  # Check if all fields are filled
    for column in st.session_state.df.columns:
        if column == 'Patient ID':
            new_record[column] = st.text_input(f"Enter {column} (must be unique)", key=f"add_{column}")
        elif st.session_state.df[column].dtype == 'object':
            new_record[column] = st.text_input(f"Enter {column}", key=f"add_{column}")
        elif st.session_state.df[column].dtype == 'int64':
            new_record[column] = st.number_input(f"Enter {column}", min_value=0, step=1, value=0, key=f"add_{column}")
        elif st.session_state.df[column].dtype == 'float64':
            new_record[column] = st.number_input(f"Enter {column}", step=0.01, value=0.0, key=f"add_{column}")

        # Check if field is empty
        if not str(new_record[column]):
            all_filled = False

    if st.button('Add Record', key="add_button"):
        if all_filled and new_record['Patient ID'] not in st.session_state.df['Patient ID'].astype(str).values:
            new_row = pd.DataFrame([new_record])
            st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
            save_to_csv()  # Save to CSV
            st.success("Record added successfully!")
        elif not all_filled:
            st.error("All fields are mandatory. Please fill in all fields.")
        else:
            st.error("Patient ID must be unique.")

# --- Tab 4: Update a Record ---
elif tabs == 'Update a Record':
    st.header('Update an Existing Record')

    patient_id_to_update = st.text_input("Enter Patient ID of the record to update", key="update_patient_id")
    if patient_id_to_update and patient_id_to_update in st.session_state.df['Patient ID'].astype(str).values:
        st.write("Edit the selected record:")
        record_to_update = st.session_state.df[st.session_state.df['Patient ID'].astype(str) == patient_id_to_update].iloc[0]
        updated_record = {}
        all_filled_update = True
        for column in st.session_state.df.columns:
            if column == 'Patient ID':
                updated_record[column] = record_to_update[column]  # Patient ID is non-editable
            elif st.session_state.df[column].dtype == 'object':
                updated_record[column] = st.text_input(f"Edit {column}", value=str(record_to_update[column]), key=f"update_{column}")
            elif st.session_state.df[column].dtype == 'int64':
                updated_record[column] = st.number_input(f"Edit {column}", min_value=0, step=1, value=int(record_to_update[column]), key=f"update_{column}")
            elif st.session_state.df[column].dtype == 'float64':
                updated_record[column] = st.number_input(f"Edit {column}", step=0.01, value=float(record_to_update[column]), key=f"update_{column}")
            
            # Check if field is empty
            if not str(updated_record[column]):
                all_filled_update = False

        if st.button("Update Record", key="update_button"):
            if all_filled_update:
                for key, value in updated_record.items():
                    st.session_state.df.loc[st.session_state.df['Patient ID'] == record_to_update['Patient ID'], key] = value
                save_to_csv()  # Save updated data
                st.success("Record updated successfully!")
            else:
                st.error("All fields are mandatory. Please fill in all fields.")
    elif patient_id_to_update:
        st.error("Patient ID not found.")

# --- Tab 5: Delete a Record ---
elif tabs == 'Delete Record':
    st.header('Delete a Record')

    # Input field for Patient ID to delete
    patient_id_to_delete = st.text_input("Enter Patient ID of the record to delete", key="delete_patient_id")
    
    # Check if Patient ID is found in the dataset
    if patient_id_to_delete and patient_id_to_delete in st.session_state.df['Patient ID'].astype(str).values:
        # Show the record to be deleted
        record_to_delete = st.session_state.df[st.session_state.df['Patient ID'].astype(str) == patient_id_to_delete].iloc[0]
        st.write("Record to be deleted:")
        st.write(record_to_delete)

        # Confirm deletion
        if st.button('Delete Record', key="delete_button"):
            # Delete the record from the DataFrame (in-memory update)
            st.session_state.df = st.session_state.df[st.session_state.df['Patient ID'].astype(str) != patient_id_to_delete].reset_index(drop=True)
            
            # Immediately update the CSV file to persist the change
            save_to_csv()
            
            # Show success message
            st.success(f"Record with Patient ID {patient_id_to_delete} deleted successfully!")
            st.session_state.df = pd.read_csv('csvcleaned_heart_attack_prediction_dataset.csv')
    elif patient_id_to_delete:
        st.error("Patient ID not found.")
