import streamlit as st
import pickle
import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open("vec.pkl", 'rb') as file:
    vec = pickle.load(file)
artciles = pd.read_csv('merged.csv')
artciles_vec = vec.transform(artciles['Title'])   

label_map = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech",
    
}



def recommend_articles(input_text, vec, article_vec, articles, top_n=5):
    input_vec = vec.transform([input_text])
    similarities = cosine_similarity(input_vec, article_vec).flatten()
    top_indices = similarities.argsort()[::-1]
    filtered_indices = [i for i in top_indices if similarities[i] < 0.999][:top_n]
    results = articles.iloc[filtered_indices][['Class Index', 'text']].copy()
    results['Category'] = results['Class Index'].map(label_map)
    return results[['Category', 'text']]



st.title("News Recommendation System")


user_input = st.text_input("Enter a news headline or text to get recommendations:")

if user_input:
    input_vec = vec.transform([user_input])
    pred=model.predict(input_vec)[0]
    st.write(f"Your Article Category: {label_map[pred]}")
    st.write("Recommended Articles:")
    recommendations = recommend_articles(user_input, vec, artciles_vec, artciles)   

    for i, row in recommendations.iterrows():
        st.write(f"**Category:** {row['Category']   }")
        st.write(f"**Article:** {row['text']}")
else:  
    st.write("Please enter a news headline or text to get recommendations.")
    



