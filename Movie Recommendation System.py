import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Step 1: Load Dataset
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', usecols=[0, 1], names=['item_id', 'title'])
data = pd.merge(ratings, movies, on='item_id')

# Step 2: Create User-Item Matrix
user_item_matrix = data.pivot_table(index='user_id', columns='title', values='rating')
user_item_filled = user_item_matrix.fillna(0)

# Step 3: Compute User Similarity
user_similarity = cosine_similarity(user_item_filled)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_filled.index, columns=user_item_filled.index)

# Step 4: User-Based Recommendation
def recommend_user_based(user_id, user_item_matrix, similarity_matrix, top_n=5):
    sim_scores = similarity_matrix.loc[user_id]
    similar_users = sim_scores.sort_values(ascending=False).index[1:]  # exclude self
    weighted_scores = {}
    for other_user in similar_users:
        weight = sim_scores.loc[other_user]
        other_user_ratings = user_item_matrix.loc[other_user]
        for movie, rating in other_user_ratings.dropna().items():
            if pd.isna(user_item_matrix.loc[user_id, movie]):
                weighted_scores[movie] = weighted_scores.get(movie, 0) + weight * rating
    sorted_movies = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
    return [movie for movie, _ in sorted_movies[:top_n]]

# Bonus 1: Item-Based Filtering
item_similarity = cosine_similarity(user_item_filled.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

def recommend_item_based(movie_title, item_similarity_matrix, top_n=5):
    similar_scores = item_similarity_matrix[movie_title].sort_values(ascending=False)[1:top_n+1]
    return similar_scores.index.tolist()

# Bonus 2: SVD Recommendation
svd = TruncatedSVD(n_components=20)
latent_matrix = svd.fit_transform(user_item_filled)
predicted_matrix = np.dot(latent_matrix, svd.components_)
predicted_df = pd.DataFrame(predicted_matrix, index=user_item_filled.index, columns=user_item_filled.columns)

def recommend_svd(user_id, pred_df, user_item_matrix, top_n=5):
    user_ratings = pred_df.loc[user_id]
    seen_movies = user_item_matrix.loc[user_id].dropna().index
    recommendations = user_ratings.drop(labels=seen_movies).sort_values(ascending=False).head(top_n)
    return recommendations.index.tolist()

# Final Output
print("----- User-Based Recommendations for User 10 -----")
print(recommend_user_based(10, user_item_matrix, user_similarity_df))

print("\n----- Item-Based Recommendations for 'Star Wars (1977)' -----")
print(recommend_item_based('Star Wars (1977)', item_similarity_df))

print("\n----- SVD-Based Recommendations for User 10 -----")
print(recommend_svd(10, predicted_df, user_item_matrix))

# BONUS 3: Evaluate using Precision@K
def precision_at_k(user_id, user_item_matrix, k=5):
    user_ratings = user_item_matrix.loc[user_id]
    rated_movies = user_ratings[user_ratings > 0].index.tolist()
    if len(rated_movies) < 3:
        return None  # not enough data to evaluate

    train_movies = rated_movies[:-2]
    test_movies = rated_movies[-2:]

    temp_matrix = user_item_matrix.copy()
    for movie in test_movies:
        temp_matrix.loc[user_id, movie] = 0  # zero out test movies

    temp_user_similarity = cosine_similarity(temp_matrix.fillna(0))
    temp_user_similarity_df = pd.DataFrame(temp_user_similarity, index=temp_matrix.index, columns=temp_matrix.index)

    recommendations = recommend_user_based(user_id, temp_matrix, temp_user_similarity_df, top_n=k)

    hits = len(set(recommendations) & set(test_movies))
    precision = hits / k
    return precision

# Show precision@5 for User 10
precision = precision_at_k(10, user_item_matrix, k=5)
print(f"\n----- Precision@5 for User 10 -----\n{precision:.2f}" if precision is not None else "Not enough ratings for evaluation.")
