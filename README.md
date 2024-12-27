# House_Price_Prediction

# Bengaluru House Price Prediction

## 📋 Overview
This project focuses on predicting house prices in Bengaluru using machine learning techniques. The aim is to build a model that provides accurate price predictions based on input features such as location, total area (in square feet), number of bedrooms, bathrooms, and other relevant factors.

## 🚀 Features
- **Data Cleaning and Preprocessing**: Handling missing values, outliers, and non-numeric values in the dataset.
- **Exploratory Data Analysis (EDA)**: Visualizing the data to uncover trends and relationships.
- **Feature Engineering**: Creating new features such as `Price_per_Sqft` to enhance model performance.
- **Model Development**: Training and evaluating regression models to predict house prices.
- **Streamlit Dashboard**: Interactive web app to visualize data and predict house prices.

## 🛠️ Technologies Used
- **Python**: For data analysis, preprocessing, and modeling.
- **Pandas & NumPy**: For data manipulation and numerical computations.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For machine learning models and evaluation.
- **Streamlit**: For building the interactive web application.

## 📂 Project Structure
```
📁 Bengaluru_House_Price_Prediction
│
├── 📄 README.md                 # Project documentation
├── 📄 requirements.txt          # Required Python libraries
├── 📁 data                      # Dataset folder
│   └── bengaluru_house_data.csv # Raw dataset
├── 📄 house.jpg                 # Sample house image
├── 📄 house_price_prediction.ipynb  # Jupyter notebook for analysis
└── 📄 app.py                    # Streamlit application script
```

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/bengaluru-house-price-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd bengaluru-house-price-prediction
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Usage
1. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```
2. Use the app to explore the dataset and make predictions based on user inputs.

## 📈 Model Performance
- **Regression Model**: Achieved an R² score of **XX%** on the test dataset.
- **Feature Importance**: Location and total square footage are significant predictors.

## 🏡 How It Works
1. Input the location, total square footage, number of bedrooms, and bathrooms into the app.
2. The model predicts the estimated house price based on the input features.
3. Visualize data trends and model predictions through interactive charts.

## 📌 Future Enhancements
- Incorporate more features such as proximity to schools, hospitals, and transport facilities.
- Deploy the model on cloud platforms like AWS, GCP, or Azure for scalability.
- Use advanced machine learning algorithms like Gradient Boosting or Neural Networks.

## 👨‍💻 Author
**Manikandan M**  
