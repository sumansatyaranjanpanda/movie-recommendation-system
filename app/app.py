import streamlit as st
import pandas as pd

from src.recommender import train_svd_model,get_top_n_recommendations
from src.data_loader import load_movielens_100k


def load_data():
    df=load_movielens_100k()
    model,_=train_svd_model()
    return df,model

df,model=load_data()

st.title("🎬 Movie Recommendation System")
st.write("Select a user to get personalized movie recommendations.")

# Get unique users from the dataset
user_ids = df['user_id'].unique()
user_id = st.selectbox("Select User ID", user_ids)


if st.button("Show Recommendations"):
    recommendations = get_top_n_recommendations(model, df, user_id, n=5)

    st.subheader(f"Top 5 Recommendations for User {user_id}")
    st.table(pd.DataFrame(recommendations, columns=["Movie", "Predicted Rating"]))