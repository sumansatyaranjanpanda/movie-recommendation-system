from surprise import SVD
from surprise import Dataset,Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd


from src.data_loader import load_movielens_100k

def train_svd_model():
    # Step 1: Load data using previous loader
    df = load_movielens_100k()

    # prepare data for surprise formate
    reader=Reader(rating_scale=(1,5))
    data=Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

    # Step 3: Split into training and testing
    trainset, testset = train_test_split(data, test_size=0.2)

    model=SVD()
    model.fit(trainset)


    # Step 5: Evaluate
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)

    return model, predictions

def get_top_n_recommendations(model, df, user_id, n=10):
    # Get all items the user has already rated
    seen_items = df[df['user_id'] == user_id]['item_id'].tolist()

    # Get all unique items in dataset
    all_items = df['item_id'].unique()

    # Predict ratings for unseen items
    unseen = [item for item in all_items if item not in seen_items]
    predictions = [model.predict(user_id, item_id) for item_id in unseen]

    # Sort by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Get top-N items with title
    top_n = predictions[:n]
    top_n_items = [(pred.iid, pred.est) for pred in top_n]

    # Map item_id → title
    id_to_title = dict(zip(df['item_id'], df['title']))
    top_n_with_titles = [(id_to_title[item_id], round(score, 2)) for item_id, score in top_n_items]

    return top_n_with_titles


if __name__ == '__main__':
    model, predictions = train_svd_model()
    df = load_movielens_100k()

    user_id = 10  # Change to test other users
    recommendations = get_top_n_recommendations(model, df, user_id, n=5)

    print(f"\nTop 5 recommendations for User {user_id}:")
    for title, score in recommendations:
        print(f"{title} (Predicted Rating: {score})")

