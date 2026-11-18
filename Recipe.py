
import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Recipe Recommender",
    page_icon="🍲",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #F8F8F2;
}

.block-container {
    padding: 2rem 5rem 3rem 5rem;
    max-width: 1400px;
}

header {
    background: transparent !important;
}

.hero {
    background: linear-gradient(120deg, #153D2B, #286044);
    padding: 55px 50px;
    border-radius: 24px;
    color: white;
    margin-bottom: 35px;
}

.hero h1 {
    color: white;
    font-size: 43px;
    font-weight: 800;
    margin-bottom: 12px;
}

.hero p {
    color: #DCE9DF;
    font-size: 17px;
    margin-bottom: 0;
}

.section-title {
    color: #193D2B;
    font-size: 25px;
    font-weight: 800;
    margin: 25px 0 15px 0;
}

.search-box {
    background: white;
    padding: 28px;
    border-radius: 18px;
    border: 1px solid #E6E9E1;
    margin-bottom: 25px;
}

div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    border-radius: 10px;
}

.stButton > button {
    background: #1D5538;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 25px;
    font-weight: 700;
    width: 100%;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #286C48;
    color: white;
    border: none;
}

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #E6E9E1;
    text-align: center;
}

.metric-label {
    color: #718071;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #1D5538;
    font-size: 30px;
    font-weight: 800;
    margin-top: 5px;
}

.recipe-card {
    background: white;
    border: 1px solid #E5E9E1;
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 15px;
    min-height: 155px;
    box-shadow: 0 3px 12px rgba(20, 50, 30, 0.04);
}

.recipe-card h4 {
    color: #193D2B;
    font-size: 17px;
    margin-bottom: 12px;
}

.recipe-detail {
    color: #68766A;
    font-size: 13px;
    margin-bottom: 8px;
}

.similarity {
    color: #1D7045;
    font-weight: 800;
    font-size: 14px;
}

.footer {
    text-align: center;
    color: #829083;
    font-size: 13px;
    padding: 30px 0 10px 0;
}

</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>🍲 Discover Your Next Favourite Meal</h1>
    <p>Explore recipes and find dishes similar to the ones you already love.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- MODEL ----------------

recipe_features = ['minutes', 'n_steps', 'n_ingredients']


@st.cache_resource
def load_model():

    base_dir = Path(__file__).resolve().parent

    data_path = base_dir / "data" / "RAW_recipes.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {data_path}"
        )

    data = pd.read_csv(data_path)

    data = (
        data.dropna(subset=['id', 'name'])
        .drop_duplicates(subset=['id'])
        .reset_index(drop=True)
    )

    for col in recipe_features:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        data[col] = data[col].fillna(data[col].median())

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data[recipe_features])

    knn = NearestNeighbors(
        metric='cosine',
        algorithm='brute',
        n_neighbors=11,
        n_jobs=-1
    )

    knn.fit(scaled)

    return data, scaled, knn


df, features_scaled, model = load_model()


# ---------------- SEARCH SECTION ----------------

st.markdown(
    '<div class="section-title">Find a Recipe</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="search-box">', unsafe_allow_html=True)

recipe_name = st.text_input(
    "Search by recipe name",
    placeholder="Example: Arrabbiata, Pasta, Chocolate Cake..."
)

if recipe_name:

    matches = df[
        df['name'].str.contains(
            recipe_name,
            case=False,
            na=False,
            regex=False
        )
    ]

    if matches.empty:

        st.warning("No matching recipes found. Try another recipe name.")

    else:

        options = {
            f"{row['name']} (ID: {row['id']})": index
            for index, row in matches.iterrows()
        }

        selected_recipe = st.selectbox(
            "Select your recipe",
            list(options.keys())
        )

        if st.button("✨ Find Similar Recipes"):

            recipe_index = options[selected_recipe]

            recipe = df.iloc[recipe_index]

            st.markdown(
                '<div class="section-title">Your Selected Recipe</div>',
                unsafe_allow_html=True
            )

            st.success(f"Showing recipes similar to {recipe['name']}")

            # Recipe statistics

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">COOKING TIME</div>
                    <div class="metric-value">{recipe['minutes']}</div>
                    <div class="metric-label">Minutes</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">INGREDIENTS</div>
                    <div class="metric-value">{recipe['n_ingredients']}</div>
                    <div class="metric-label">Ingredients</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">PREPARATION STEPS</div>
                    <div class="metric-value">{recipe['n_steps']}</div>
                    <div class="metric-label">Steps</div>
                </div>
                """, unsafe_allow_html=True)

            # ---------------- RECOMMENDATIONS ----------------

            distances, indices = model.kneighbors(
                features_scaled[recipe_index].reshape(1, -1),
                n_neighbors=11
            )

            recommended_indices = indices[0][1:]
            recommended_distances = distances[0][1:]

            recommendations = df.iloc[
                recommended_indices
            ][['name', 'minutes', 'n_ingredients']].copy()

            recommendations['Similarity'] = (
                (1 - recommended_distances) * 100
            ).round(2)

            st.markdown(
                '<div class="section-title">🍽️ Recommended Recipes</div>',
                unsafe_allow_html=True
            )

            st.caption(
                "Here are 10 recipes with similar characteristics."
            )

            # Display recommendations as cards

            for i in range(0, len(recommendations), 2):

                cols = st.columns(2)

                for j in range(2):

                    if i + j < len(recommendations):

                        row = recommendations.iloc[i + j]

                        with cols[j]:

                            st.markdown(f"""
                            <div class="recipe-card">
                                <h4>🍴 {row['name']}</h4>
                                <div class="recipe-detail">
                                    ⏱️ Cooking time: {row['minutes']} minutes
                                </div>
                                <div class="recipe-detail">
                                    🥗 Ingredients: {row['n_ingredients']}
                                </div>
                                <div class="similarity">
                                    ✦ {row['Similarity']}% Similarity
                                </div>
                            </div>
                            """, unsafe_allow_html=True)


st.markdown('</div>', unsafe_allow_html=True)


# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
    <p>🍲 Recipe Recommender</p>
    <p>Powered by Python · Pandas · Scikit-learn · Streamlit</p>
</div>
""", unsafe_allow_html=True)
