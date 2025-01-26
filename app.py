from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from data.download import preprocessing
from models.gnn import Model



# Initialize the FastAPI app
app = FastAPI()

data_size='big' # big or small

# Preprocess data and load necessary information
print("Preprocessing data...")
(num_users, num_movies, movie_feat, train_data, val_data, test_data,
 train_pos_edge_index, val_pos_edge_index, test_pos_edge_index,
 genre_names, movie_id_to_name) = preprocessing(data_size=data_size)  # Now returns movie_id_to_name

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
print("Loading the model...")
model = Model(
    num_users=num_users,
    num_movies=num_movies,
    num_movie_features=movie_feat.size(1),
    hidden_channels=32
)
model.load_state_dict(torch.load(f"saved_model/bpr_model_{data_size}.pth", map_location=device))
model = model.to(device)
model.eval()

# Define request and response models
class RecommendationRequest(BaseModel):
    user_id: int
    top_k: int = 10

class RecommendationResponse(BaseModel):
    movie_id: int
    name: str
    genres: str
    score: float

# Helper function for recommendations
def recommend_movies(model, full_data, user_id, movie_feat, genre_names, movie_id_to_name, top_k=10):
    """
    Recommends movies for a given user based on their preferences.

    Args:
        model (Model): The trained model.
        full_data (HeteroData): The full heterogenous graph data.
        user_id (int): The ID of the user to recommend movies for.
        movie_feat (Tensor): Tensor containing movie features.
        genre_names (List[str]): List of genre names corresponding to one-hot encoded indices.
        movie_id_to_name (dict): Mapping from movie IDs to movie names.
        top_k (int): The number of top recommendations to return.

    Returns:
        List[Dict]: A list of dictionaries containing movie details (ID, name, genres, and normalized score).
    """
    model.eval()

    # Ensure the model and data are on the same device
    for node_type in full_data.node_types:
        full_data[node_type].node_id = full_data[node_type].node_id.to(device)
        if "x" in full_data[node_type]:  # If node features exist
            full_data[node_type].x = full_data[node_type].x.to(device)

    for edge_type in full_data.edge_types:
        full_data[edge_type].edge_index = full_data[edge_type].edge_index.to(device)

    # Get the embeddings for all users and movies
    with torch.no_grad():
        x_dict = model(full_data)
        user_embs = x_dict["user"]
        movie_embs = x_dict["movie"]

    # Retrieve the embedding for the given user
    user_emb = user_embs[user_id]

    # Compute scores for all movies
    scores = model.score(user_emb.repeat(movie_embs.size(0), 1), movie_embs)

    # Min-max normalize the scores
    min_score = scores.min().item()
    max_score = scores.max().item()
    normalized_scores = (scores - min_score) / (max_score - min_score)

    # Sort movies by normalized score in descending order
    _, top_movie_indices = torch.topk(normalized_scores, top_k)

    # Generate recommendations
    recommendations = []
    for mapped_idx in top_movie_indices.cpu().numpy():
        # Decode genres using the movie_feat and genre_names
        genres = [genre_names[i] for i, val in enumerate(movie_feat[mapped_idx].tolist()) if val > 0]

        # Map the `mapped_idx` back to `movieId` for the name lookup
        movie_id = list(movie_id_to_name.keys())[mapped_idx]
        movie_name = movie_id_to_name.get(movie_id, f"Unknown Movie (ID: {mapped_idx})")

        # Append recommendation
        recommendations.append({
            "movie_id": int(movie_id),  # Use the actual movie ID
            "name": movie_name,
            "genres": ", ".join(genres) if genres else "Unknown",
            "score": float(normalized_scores[mapped_idx])
        })

    return recommendations



# API endpoint for recommendations
@app.post("/recommend/", response_model=list[RecommendationResponse])
def recommend_movies_api(request: RecommendationRequest):
    """
    Recommend movies to a user based on their preferences.
    """
    user_id = request.user_id
    top_k = request.top_k

    if user_id >= num_users:
        raise HTTPException(status_code=404, detail="User ID not found.")

    # Generate recommendations
    recommendations = recommend_movies(
        model, train_data, user_id, movie_feat, genre_names, movie_id_to_name, top_k
    )

    if not recommendations:
        raise HTTPException(status_code=404, detail="No recommendations found.")

    # Return the recommendations as a list of response models
    return [RecommendationResponse(**rec) for rec in recommendations]

# Run the app using Uvicorn
# Command to run: uvicorn app:app --reload
