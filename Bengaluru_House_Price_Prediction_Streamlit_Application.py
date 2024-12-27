import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import scipy.stats as stats
import re
from PIL import Image

# Load the dataset
@st.cache_resource
def load_data():
    House_data = pd.read_csv('Bengaluru_House_Data.csv')  
    return House_data

House_data = load_data()

# Load the image
image = Image.open('House_Price_Prediction.jpg')

# Display the image
st.image(image, caption='🏡💰House Price Prediction', use_column_width=True)

# Sidebar navigation
st.sidebar.title('☰ Menu')
options =  ['🏠🌟Home', '📥🗂️ Loading and Import', '🧩 Visualization', '🎯 Prediction']
selected_option = st.sidebar.selectbox('Select Option', options)
st.sidebar.write(f"**Selected Option:** {selected_option}")

if selected_option == '🏠🌟Home':
    
    # Display information
    st.title('🏠🌟**Home**')
    st.title('Bengaluru House Price Prediction')
    
    # Dataset link
    st.write("### 📁 Dataset Link:")  
    st.write("The dataset used in this project can be found [📁Bengaluru House Price Prediction](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data)")
    
    st.write("### 🚀Objective:")
    st.write("- Predict House prices in Bengaluru based on various features such as Total Square Feet, Number of Bathrooms, Balconies, and Location.")
    
    st.write("### 🛠️How it works:")
    st.write("- This application uses a Machine Learning Model (RandomForestRegressor) trained on historical House price data to make Predictions.")
    
    st.write("### 👩🏻‍💻Technologies Used:")
    st.write("* 🐍🧩**Python**: Programming language for Data Manipulation and Model building.")
    st.write("* 🌟🚀**Streamlit**: Framework for Building Interactive Web Applications.")
    st.write("* 🐼🔍 **Pandas**: Library for Data Manipulation and Analysis.")
    st.write("* 🤖📚**Scikit-learn**: Library for Machine Learning Algorithms.")
    st.write("* 🌡️📉 **Matplotlib** & **Seaborn**: Libraries for Data Visualization.")
    
    st.write('###  🗒️🖋️Note:')
    st.write('* Due to Dataset Limitations, the Model may not be accurate for all inputs.')
    st.write('* This is for Educational purposes only and should not be used as a real House Price Prediction system.')
    st.write('* Please use this model as a reference.')
    
    linkedin_icon = "https://cdn-icons-png.flaticon.com/512/174/174857.png"
    github_icon = "https://cdn-icons-png.flaticon.com/512/25/25231.png"

    st.markdown(
    f"""
    <div style="text-align: center; font-family: Arial, sans-serif;">
        <h3 style="color: #4CAF50;">Crafted with passion by Manikandan M</h3>
        <p>Follow me on:</p>
        <p>
            <a href="https://www.linkedin.com/in/manikandan-m-60878729a" target="_blank">
                <img src="{linkedin_icon}" width="30" style="vertical-align: middle; margin-right: 5px;">
                LinkedIn
            </a>
        </p>
        <p>
            <a href="https://github.com/Manikandan200355" target="_blank">
                <img src="{github_icon}" width="30" style="vertical-align: middle; margin-right: 5px;">
                GitHub
            </a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
    )
    
elif selected_option == '📥🗂️ Loading and Import':
    st.title('**📥🗂️ Loading and Import**')
    
    # Display initial data inspection outputs
    st.subheader('Dataset Overview')
    st.write('**Head Part of the dataset**')
    st.write(House_data.head())
    
    st.write('**Shape of the Dataset**')
    st.write(House_data.shape)
    
    st.write('**Statistical Summary**')
    st.write(House_data.describe())
    
    # Check for missing values
    st.write('**Missing Values**')
    st.write(House_data.isnull().sum())
    
    linkedin_icon = "https://cdn-icons-png.flaticon.com/512/174/174857.png"
    github_icon = "https://cdn-icons-png.flaticon.com/512/25/25231.png"

    st.markdown(
    f"""
    <div style="text-align: center; font-family: Arial, sans-serif;">
        <h3 style="color: #4CAF50;">Crafted with passion by Manikandan M</h3>
        <p>Follow me on:</p>
        <p>
            <a href="https://www.linkedin.com/in/manikandan-m-60878729a" target="_blank">
                <img src="{linkedin_icon}" width="30" style="vertical-align: middle; margin-right: 5px;">
                LinkedIn
            </a>
        </p>
        <p>
            <a href="https://github.com/Manikandan200355" target="_blank">
                <img src="{github_icon}" width="30" style="vertical-align: middle; margin-right: 5px;">
                GitHub
            </a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
    )
    
elif selected_option == '🧩 Visualization':
    st.title('🧩 Visualization')
    # Data preprocessing for visualization
    House_data['bhk'] = House_data['size'].str.split(' ').str[0].astype(int)
    House_data.drop_duplicates(inplace=True)
    House_data.dropna(subset=['location', 'bhk', 'bath'], inplace=True)
    House_data['total_sqft'] = House_data['total_sqft'].apply(
        lambda x: np.mean([float(i) for i in str(x).split('-')]) if '-' in str(x) 
        else float(re.findall(r'[0-9.]+', str(x))[0]) if re.findall(r'[0-9.]+', str(x)) else np.nan
    )
    House_data.dropna(subset=['total_sqft'], inplace=True)
    House_data['Price_per_Sqft'] = House_data['price'] * 100000 / House_data['total_sqft']

    st.write('**Scatter plot for Square Feet Vs Price**')
    plt.figure(figsize=(10, 6))
    plt.scatter(House_data['total_sqft'], House_data['price'])
    plt.title('Price vs Total Square Feet')
    plt.xlabel('Total Square Feet')
    plt.ylabel('Price')
    st.pyplot(plt.gcf())

    st.write('**Heatmap of Correlation**')
    correlation_matrix = House_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    st.pyplot(plt.gcf())

elif selected_option == '🎯 Prediction':
    st.title('🎯 **Prediction**')

    # Dropping the Unwanted Columns
    # House_data.drop(columns=['availability','society'],inplace=True)

    # Data Mapping
    Data_Mapping = {'Super built-up  Area': '1', 'Built-up  Area': '2', 'Plot  Area': '3', 'Carpet  Area': '4'}
    House_data['area_type_Numeric'] = House_data['area_type'].map(Data_Mapping)

    # Extracting Digits from the String
    House_data['bhk'] = House_data['size'].str.split(' ').str.get(0)

    # Dropping Duplicates
    House_data.drop_duplicates(inplace=True)

    # Dropping null values which is very small compared to the dataset
    House_data.dropna(subset=['location', 'bhk', 'bath'], inplace=True)

    # Datatype Conversion
    House_data['area_type_Numeric'] = pd.to_numeric(House_data['area_type_Numeric'], errors='coerce')
    House_data['bhk'] = House_data['bhk'].astype('int64')

    # Filling null values with their median
    Balcony_median = House_data['balcony'].median()
    House_data['balcony'] = House_data['balcony'].fillna(value=Balcony_median)

    # Function to process total_sqft values
    import re

    def process_total_sqft(value):
        if isinstance(value, str):
            if '-' in value:
                values = list(map(float, value.split('-')))
                return sum(values) / len(values)
            elif re.search(r'\d+', value):
                return float(re.search(r'\d+', value).group().replace('.', ''))
        try:
            return float(value)
        except ValueError:
            return None

    # Apply the function to the total_sqft column
    House_data['total_sqft'] = House_data['total_sqft'].apply(process_total_sqft)

    # Feature Extraction and Dimensionality Reduction
    df = House_data.copy()

    # In Real-Estate, Price Per Square Feet is important
    df['Price_per_Sqft'] = df['price'] * 100000 / df['total_sqft']

    # Now we need to handle the 'location' attribute
    location_unique = df.location.value_counts()

    location_unique_less_than_10 = location_unique[location_unique <= 10]

    df['location'] = df['location'].apply(lambda x: 'other' if x in location_unique_less_than_10 else x)

    # Numerical Columns
    numerical_cols = df[['bhk', 'total_sqft', 'bath', 'balcony', 'price']]

    # Z-Score to remove outliers
    from scipy import stats
    z_scores = stats.zscore(numerical_cols)
    threshold = 3
    numerical_cols = numerical_cols[(z_scores <= threshold).all(axis=1)]

    # Dropping unwanted columns from df
    df = df.drop(columns=['area_type', 'size', 'Price_per_Sqft', 'bhk', 'total_sqft', 'bath', 'balcony', 'price', 'availability', 'society'], axis=1)

    # Concatenation of two columns
    df = pd.concat([df, numerical_cols], axis=1)

    # Creating dummies for the 'location' attribute
    dummies = pd.get_dummies(df['location'])
    dummies = dummies.drop(columns=['other'], axis=1)
    dummies = dummies.astype('int64')

    # Concatenating df and dummies
    df1 = pd.concat([df, dummies], axis=1)

    # Dropping Location attribute from df1
    df1 = df1.drop(columns=['location'], axis=1)

    # Filling null values with their medians
    bhk = df['bhk'].median()
    total_sqft = df['total_sqft'].median()
    bath = df['bath'].median()
    balcony = df['balcony'].median()
    price = df['price'].median()
    df1.bhk = df1.bhk.fillna(value=bhk)
    df1.total_sqft = df1.total_sqft.fillna(value=total_sqft)
    df1.bath = df1.bath.fillna(value=bath)
    df1.balcony = df1.balcony.fillna(value=balcony)
    df1.price = df1.price.fillna(value=price)

    # Splitting the data
    X = df1.drop(columns=['price'], axis=1)
    y = df1['price']

    # Split the data into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the StandardScaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform both training and test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the Linear Regression model
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test_scaled)

    # Model evaluation
    from sklearn.metrics import r2_score
    model_score = model.score(X_train_scaled, y_train)
    test_score = r2_score(y_test, predictions)

    # User input for prediction
    st.header('Input Features for Prediction')
    total_sqft = st.number_input('Total Square Feet', min_value=0.0, step=1.0)
    bath = st.number_input('Number of Bathrooms', min_value=1, step=1)
    balcony = st.number_input('Number of Balconies', min_value=0, step=1)
    location = st.selectbox('Location', options=dummies.columns)

    # Prepare input for prediction
    input_features_df = pd.DataFrame(np.zeros((1, X.shape[1])), columns=X.columns)
    input_features_df['total_sqft'] = total_sqft
    input_features_df['bath'] = bath
    input_features_df['balcony'] = balcony

    if location in dummies.columns:
            loc_index = np.where(dummies.columns == location)[0][0]
            input_features_df.iloc[0, X.columns.get_loc(location)] = 1

    # When button is clicked, make prediction
    if st.button('Predict'):
        prediction = model.predict(input_features_df)[0]  # Define prediction here
        st.write(f"Predicted House Price: ₹{prediction:,.2f}")

        # Prepare the results for download
        prediction_df = pd.DataFrame({
            'Total_Sqft': [total_sqft],
            'Bathrooms': [bath],
            'Balconies': [balcony],
            'Location': [location],
            'Predicted Price (₹)': [prediction]
        })

        # Create a CSV file for download
        csv = prediction_df.to_csv(index=False)
        st.download_button(
            label="Download Prediction as CSV",
            data=csv,
            file_name='house_price_prediction.csv',
            mime='text/csv'
        )
     # Social links
      linkedin_icon = "https://cdn-icons-png.flaticon.com/512/174/174857.png"
      github_icon = "https://cdn-icons-png.flaticon.com/512/25/25231.png"

      st.markdown(
        f"""
        <div style="text-align: center; font-family: Arial, sans-serif;">
            <h2 style="color: #4CAF50;">Crafted with passion by Manikandan M</h2>
            <p>Follow me on:</p>
            <p>
                <a href="https://www.linkedin.com/in/manikandan-m-60878729a" target="_blank">
                    <img src="{linkedin_icon}" width="30" style="vertical-align: middle; margin-right: 5px;">
                    LinkedIn
                </a>
            </p>
            <p>
                <a href="https://github.com/Manikandan200355" target="_blank">
                    <img src="{github_icon}" width="30" style="vertical-align: middle; margin-right: 5px;">
                    GitHub
                </a>
            </p>
        </div>
        """,
        unsafe_allow_html=True
         )
