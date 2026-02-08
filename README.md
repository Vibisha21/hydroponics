🌱 Smart Hydroponic Nutrient & Cultivation Predictor

An AI-powered Streamlit web application that predicts optimal cultivation duration and nutrient requirements for hydroponic crops based on environmental and crop parameters.
This project uses machine learning to assist growers in making data-driven decisions for efficient hydroponic farming.


**Features**
- Predicts **cultivation duration (days)**
- Estimates **nutrient requirements**:
  - Nitrogen (N)
  - Phosphorus (P)
  - Potassium (K)
  - Calcium (Ca)
  - Magnesium (Mg)
- Supports multiple crop types and growth stages
- Interactive UI built with **Streamlit**
- Visual nutrient breakdown using charts
- Model trained dynamically at runtime (Streamlit Cloud–friendly)


**Tech Stack**
- Python
- Streamlit – frontend & deployment
- Pandas, NumPy – data handling
- Scikit-learn – machine learning
- Matplotlib – visualization


**Dataset**
The model is trained on a curated dataset containing hydroponic crop information, including:
- Plant type
- Growth stage
- Environmental parameters (temperature, humidity, light intensity)
- Nutrient requirements

Dataset file:
Hydroponics_Indian_Plants_1000.csv

**⚙️ How It Works**
1. User selects crop type, growth stage, and environmental conditions.
2. The application trains a machine learning model at runtime.
3. The trained model predicts cultivation days and nutrient needs.
4. Results are displayed numerically and visually.

**Running Locally**
```bash
pip install -r requirements.txt
streamlit run app.py
```

hydroponics/
- │── app.py
- │── train_model.py
- │── Hydroponics_Indian_Plants_1000.csv
- │── requirements.txt
- │── README.md


**Author**
- Vibisha V
- AI & Data Science Student
- Interested in Machine Learning and Data Analytics.
