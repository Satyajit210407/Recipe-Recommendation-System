
# 🍲 Recipe Recommendation System

A Machine Learning-based Recipe Recommendation System that helps users discover similar recipes based on cooking time, number of ingredients, and preparation steps.

Built using Python, Scikit-learn, and Streamlit.

## ✨ Features

- Search recipes by name.
- Select a recipe from the search results.
- Get 10 similar recipe recommendations.
- View cooking time, ingredients, and preparation steps.
- Display similarity scores for recommended recipes.
- Interactive Streamlit web interface.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- StandardScaler
- Nearest Neighbors
- Cosine Similarity

## 🧠 Machine Learning Workflow

1. Data Loading
2. Data Cleaning and Preprocessing
3. Feature Selection
4. Feature Scaling using StandardScaler
5. Model Training using Nearest Neighbors
6. Similarity Calculation using Cosine Distance
7. Recipe Recommendations

## 📊 Dataset

**Food.com Recipes and User Interactions**

Dataset Source: [Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)

Download `RAW_recipes.csv` and place it inside the `data` folder.

## 📁 Project Structure

```text
Recipe-Recommendation-System/
│
├── .gitignore
├── Recipe.py
├── Recipe_Recommendation_System.ipynb
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Recipe-Recommendation-System.git
```

### 2. Navigate to the project directory

```bash
cd Recipe-Recommendation-System
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place `RAW_recipes.csv` inside the `data` folder.

### 5. Run the application

```bash
streamlit run Recipe.py
```

## 🖥️ Application Preview

Add your Streamlit application screenshot here.

## 👨‍💻 Author

**Satyajit Pradhan**

- [GitHub](https://github.com/Satyajit210407)
- [LinkedIn](https://www.linkedin.com/in/satyajitpradhan-21x04x07/)

---

⭐ If you like this project, consider giving it a star!
