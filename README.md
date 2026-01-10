
# 🧑‍⚕️ Doctor At Home 🩺

**Doctor At Home** is an AI-powered medical diagnosis assistant built using **Flask**, **Machine Learning**, and **Google Gemini AI**. Based on user-selected symptoms, the app predicts possible diseases using three ML models (Decision Tree, Random Forest, Naive Bayes) and provides AI-generated insights, medical precautions, and specialist suggestions.


## 🚀 Features

- 🧠 **Disease Prediction** using:
  - Decision Tree
  - Random Forest
  - Naive Bayes
- 🤖 **Google Gemini AI Integration** for detailed medical insights
- 🧾 **Precautionary Measures** & **Recommended Specialists**
- 🖥️ Clean and interactive web interface for symptom selection
- ✅ Works with `.csv` datasets for training/testing
- ⚠ Disclaimer message for non-professional diagnosis

---

## 🗂️ Project Structure

├── app.py                      # Main Flask app
├── Training.csv                # Training data
├── Testing.csv                 # Testing data
├── templates/
│   └── index.html              # Web UI template
├── .env                        # Environment variables (e.g., GEMINI\_API\_KEY)
├── static/                     # Static files (optional)
└── README.md

````

---

## ⚙️ Setup Instructions

### 1. 🔧 Install Dependencies

```bash
pip install flask scikit-learn pandas numpy python-dotenv google-generativeai
````

### 2. 🔑 Add Gemini API Key

Create a `.env` file in the root directory:

```
GEMINI_API_KEY=your_google_gemini_api_key
```

### 3. ▶️ Run the App

```bash
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000/`.

---

## 💡 How It Works

1. **Input Symptoms**: Users select symptoms from a checklist.
2. **Prediction Models**:

   * Three ML models predict the most probable disease.
3. **Precaution Info**:

   * Disease-specific precautions and recommended doctors are shown.
4. **Gemini AI Insights**:

   * Gemini AI provides causes, treatments, and further details for the predicted disease.

---

## 🧠 Models Used

* `DecisionTreeClassifier`
* `RandomForestClassifier`
* `GaussianNB`

All models are trained on `Training.csv` and validated using `Testing.csv`.

---

## 🌐 Routes

### `/`

* Main page to input symptoms and view predictions

### `/predict` (POST)

* Handles form submission
* Performs predictions and displays results with AI response


## 📋 Technologies Used

* Flask
* Pandas & NumPy
* Scikit-learn
* Google Generative AI (Gemini)
* HTML + Jinja2 Templates

## 📄 License

Licensed under the [MIT License](LICENSE)


