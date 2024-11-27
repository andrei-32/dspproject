import os
import warnings
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tempfile import NamedTemporaryFile

# Suppress TensorFlow and Streamlit warnings
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")

# Set the environment variable for TensorFlow optimization (optional)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Flower names for classification
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Flower care and general information
flower_info = {
    'daisy': {
        'care': 'Daisies thrive in full sun with moderate watering. They prefer well-drained soil.',
        'general': 'Daisies are cheerful flowers that are easy to grow. They are often symbols of purity and innocence.'
    },
    'dandelion': {
        'care': 'Dandelions are hardy and grow in most soils. They do well in full sun and require minimal watering.',
        'general': 'Dandelions are often considered weeds, but they are also highly nutritious and can be used in salads or teas.'
    },
    'rose': {
        'care': 'Roses need full sunlight, regular watering, and well-drained soil. Prune them to promote healthy growth.',
        'general': 'Roses are classic flowers that symbolize love and beauty. They come in a variety of colors and types.'
    },
    'sunflower': {
        'care': 'Sunflowers require full sun and well-drained soil. They need regular watering and space to grow tall.',
        'general': 'Sunflowers are known for their large yellow petals and are often associated with happiness and positivity.'
    },
    'tulip': {
        'care': 'Tulips prefer cool weather and well-drained soil. Water them moderately and plant them in a sunny location.',
        'general': 'Tulips are spring-blooming flowers that come in various colors, often symbolizing perfect love.'
    }
}

# Admin credentials (you can adjust these)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# Try loading the model and catch errors
try:
    model = load_model('Flower_Recog_Model.h5')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #st.write("Model loaded successfully!")
except Exception as e:
    st.write(f"Error loading model: {e}")

# Function to classify uploaded image
def classify_images(image_path):
    try:
        input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = tf.expand_dims(input_image_array, 0)
        
        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])
        
        flower_type = flower_names[np.argmax(result)]
        return flower_type, result
    except Exception as e:
        st.write(f"Error during prediction: {e}")
        return None, None

# Read inventory CSV file using pandas
def load_inventory():
    try:
        # Load inventory from CSV
        inventory_data = pd.read_csv('inventory.csv')

        # Clean Price column and convert to numeric
        inventory_data['Price'] = inventory_data['Price'].replace({r'[^\d.]': ''}, regex=True)
        inventory_data['Price'] = pd.to_numeric(inventory_data['Price'], errors='coerce')
        inventory_data['Stock'] = pd.to_numeric(inventory_data['Stock'], errors='coerce')

        # Handle NaN values
        inventory_data['Price'] = inventory_data['Price'].fillna(0.0)
        inventory_data['Stock'] = inventory_data['Stock'].fillna(0)

        return inventory_data
    except FileNotFoundError:
        st.write("Inventory file not found!")
        return pd.DataFrame(columns=["Name", "Price", "Stock"])

# Update the CSV with new price and stock
def update_inventory(flower_name, new_price, new_stock):
    try:
        inventory_data = load_inventory()
        # Find the row corresponding to the flower name
        flower_row = inventory_data[inventory_data['Name'].str.lower() == flower_name.lower()]
        
        if not flower_row.empty:
            # Update the Price and Stock values
            inventory_data.loc[inventory_data['Name'].str.lower() == flower_name.lower(), 'Price'] = new_price
            inventory_data.loc[inventory_data['Name'].str.lower() == flower_name.lower(), 'Stock'] = new_stock

            # Save the updated data back to the CSV
            inventory_data.to_csv('inventory.csv', index=False)
            st.success(f"Successfully updated {flower_name.capitalize()} with new price and stock!")
        else:
            st.warning(f"{flower_name.capitalize()} not found in inventory.")
    except Exception as e:
        st.write(f"Error updating inventory: {e}")

# Flower Classification Page
def classification_page():
    st.header('Flower Classification')
    uploaded_file = st.file_uploader('Upload an Image', type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        try:
            # Save the uploaded image to a temporary file
            with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            # Display the uploaded image
            st.image(temp_file_path, width=200)

            # Perform classification
            flower_type, result = classify_images(temp_file_path)

            if flower_type:  # If prediction was successful
                st.markdown(f"The flower is a **{flower_type}** with a confidence of **{np.max(result) * 100:.2f}%**.")

                # Display flower care and general info
                st.subheader(f"Care Tips for {flower_type.capitalize()}:")
                st.write(flower_info[flower_type]['care'])

                st.subheader(f"General Information about {flower_type.capitalize()}:")
                st.write(flower_info[flower_type]['general'])

                # Load inventory and display stock and price for the identified flower
                inventory_data = load_inventory()
                if not inventory_data.empty:
                    flower_inventory = inventory_data[inventory_data['Name'].str.lower() == flower_type.lower()]
                    
                    if not flower_inventory.empty:
                        flower_price = flower_inventory['Price'].values[0]
                        flower_stock = flower_inventory['Stock'].values[0]

                        # Format Price as float with 2 decimal places
                        st.subheader(f"Current Stock & Price for {flower_type.capitalize()}:")
                        st.write(f"Price: ₱{flower_price:.2f}")
                        st.write(f"Stock: {int(flower_stock)} units available")
                    else:
                        st.write(f"Sorry, the flower {flower_type.capitalize()} is not listed in the inventory.")
                else:
                    st.write("Error: Inventory data could not be loaded.")
            else:
                st.write("There was an issue with classification. Please try again.")
        except Exception as e:
            st.write(f"Error saving the file: {e}")

# Inventory Page with Authentication
def inventory_page():
    st.header('Flower Inventory')

    # Check if the admin is logged in
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        # Admin login form
        st.subheader('Admin Login')
        username = st.text_input('Username', key='username')
        password = st.text_input('Password', type='password', key='password')
        
        if st.button('Login'):
            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                st.session_state.logged_in = True  # Set logged_in to True in session state
                st.success('Login successful!')
            else:
                st.error("Invalid username or password. Please try again.")
    else:
        # Admin is logged in, show the inventory page
        #st.write("You are logged in as an admin.")
        
        # Logout button
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.write("You have logged out successfully.")
            st.stop()  # Stop the script to trigger a page reload

        # Display inventory data and allow updates
        inventory_data = load_inventory()

        if not inventory_data.empty:
            st.write("Current Flower Inventory:")
            st.dataframe(inventory_data)  # Display the inventory as a table using pandas

            # Edit stock and price for a selected flower
            flower_name = st.selectbox("Select Flower to Edit", inventory_data['Name'])
            new_price = st.number_input("New Price", min_value=0.0, value=float(inventory_data[inventory_data['Name'] == flower_name]['Price'].values[0]))
            new_stock = st.number_input("New Stock", min_value=0, value=int(inventory_data[inventory_data['Name'] == flower_name]['Stock'].values[0]))

            if st.button(f"Update {flower_name}"):
                update_inventory(flower_name, new_price, new_stock)
        else:
            st.write("No inventory data available.")

# Main layout
def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.radio("Choose a page", ("Flower Classification", "Inventory"))
    
    if page == "Flower Classification":
        classification_page()
    elif page == "Inventory":
        inventory_page()

if __name__ == "__main__":
    main()
