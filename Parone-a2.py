#Course: INFO-629-686 - FA 25-26
#Assignment 2: Demo of a recommender system application
#Student: Anthony Parone
#Date: October 2025
#reference: https://machinelearningmastery.com/building-a-recommender-system-from-scratch-with-matrix-factorization-in-python/




##libraries
import numpy as np #Installed Version numpy 1.26.4 - numpy needs to be < version 2
import pandas as pd #Installed Version pandas 2.3.3
import matplotlib.pyplot as plt #Installed Version matplotlib 1.5.0
from surprise import Dataset, Reader, SVD #Installed Version surprise 0.1
from surprise.model_selection import train_test_split, cross_validate
from surprise import accuracy
import requests #Installed Version requests 2.32.5
import zipfile #Standard Python Library
import io #Standard Python Library
import os #Standard Python Library

##globals
#set how many top N recommendations should be returned
recommend_count =10
#set user
user_id = 10


##functions##
def download_and_extract_movielens():
    if not os.path.exists('ml-100k'): 
        print("Downloading MovieLens 100K dataset...")
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall()
        print("Movielens 100K dataset downloaded and extracted successfully.")
    else: #assumes data files are in the directory
        print("The dataset already exists. Download skipped.")
        
def get_movie_names():
    movies_df = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None, usecols=[0, 1], names=['item_id', 'title'])
    return movies_df

def get_ratings():
    ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    return ratings_df

def recommend_movies(user_id, n=10):
    # List of all movies
    all_movies = movies_df['item_id'].unique()
    
    # Movies already rated by the user
    rated_movies = ratings_df[ratings_df['user_id'] == user_id]['item_id'].values
    
    # Movies not yet rated by the user
    unrated_movies = np.setdiff1d(all_movies, rated_movies)
    
    # Predicting ratings on unseen movies, by using the trained SVD model
    predictions = []
    for item_id in unrated_movies:
        predicted_rating = model.predict(user_id, item_id).est
        predictions.append((item_id, predicted_rating))
    
    # Rank predictions by estimated rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N recommendations
    top_recommendations = predictions[:n]
    
    # Fetch movie titles associated with top N recommendations
    recommendations = pd.DataFrame(top_recommendations, columns=['item_id', 'predicted_rating'])
    recommendations = recommendations.merge(movies_df, on='item_id')
    
    return recommendations

def display_data_summary(ratings_df):
    print(f"Data Set Description:")
    #print(f"Dataset shape: {ratings_df.shape}")
    #print(f"Dataset columns: {ratings_df.head(1)}")
    print(f"Number of unique users: {ratings_df['user_id'].nunique()}")
    print(f"Number of unique movies: {ratings_df['item_id'].nunique()}")
    print(f"Number of ratings: {len(ratings_df)}")
    print(f"Range of ratings: {ratings_df['rating'].min()} to {ratings_df['rating'].max()}")



##Application Start

#begin data load
download_and_extract_movielens()
#load movie names
movies_df = get_movie_names()
#movielens:user data
ratings_df = get_ratings()



#------------------------------------------------------------------------------------------
#recomendations by user_id - no changes to data set
##train the model and retrieve recommendations
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['user_id', 'item_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
model = SVD(n_factors=20, lr_all=0.01, reg_all=0.01, n_epochs=20, random_state=42)
model.fit(trainset)
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)
cv_results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
recommendations = recommend_movies(user_id, n=recommend_count)

#output
print(f"---------------------------------------------------------------------------")
#display dataset summary
display_data_summary(ratings_df)
print(f"Recomendations and Ratings")
print(f"\nTop 10 MovieLens Ratings for {user_id}:")
print(ratings_df.loc[ratings_df['user_id'] == user_id].nlargest(10, 'rating'))
print(f"\nTop {recommend_count} recommended movies for user {user_id}:")
print(recommendations[['title', 'predicted_rating']])


#------------------------------------------------------------------------------------------
#recomendations by user_id - all ratings for this user_id are set to the same value
new_rating = 2
#assign new ratings by user_id
ratings_df.loc[ratings_df['user_id'] == user_id, 'rating'] = new_rating

##train the model and retrieve recommendations
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['user_id', 'item_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
model = SVD(n_factors=20, lr_all=0.01, reg_all=0.01, n_epochs=20, random_state=42)
model.fit(trainset)
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)
cv_results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
updated_recommendations = recommend_movies(user_id, n=recommend_count)

#output
print(f"------------------------------------------------------------------------------------------")
#display dataset summary
display_data_summary(ratings_df)
print(f"\nTop 10 Ratings for user {user_id} after updating ratings to the value of {new_rating}:")
print(ratings_df.loc[ratings_df['user_id'] == user_id].nlargest(10, 'item_id'))
print(f"\nTop {recommend_count} recommended movies for user {user_id} after updating ratings to the value of {new_rating}:")
print(updated_recommendations[['title', 'predicted_rating']])


#------------------------------------------------------------------------------------------
#recomendations by user_id - all ratings for this user_id are deleted
indices_to_drop = ratings_df[ratings_df['user_id'] == user_id].index
ratings_df.drop(indices_to_drop, inplace=True)

##train the model and retrieve recommendations
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['user_id', 'item_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
model = SVD(n_factors=20, lr_all=0.01, reg_all=0.01, n_epochs=20, random_state=42)
model.fit(trainset)
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)
cv_results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
deleted_recommendations = recommend_movies(user_id, n=recommend_count)

#output
print(f"------------------------------------------------------------------------------------------")
#display dataset summary
display_data_summary(ratings_df)
print(f"\nTop 10 Ratings for user {user_id} after deleting their ratings:")
print(ratings_df.loc[ratings_df['user_id'] == user_id].nlargest(10, 'item_id'))
print(f"\nTop {recommend_count} recommended movies for user {user_id} after deleting ratings:")
print(deleted_recommendations[['title', 'predicted_rating']])





