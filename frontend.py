import streamlit as st
import pandas as pd
import requests

# Set the page layout to wide
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# Define the API endpoint
API_URL = "http://127.0.0.1:8000/recommend/"  # Replace with your actual FastAPI endpoint

# Streamlit app
st.title("🎬 Movie Recommendation System")

# Input for user ID
st.sidebar.header("Input User Details")
user_id = st.sidebar.number_input("Enter User ID:", min_value=0, step=1, value=0)

# Slider for number of recommendations
top_k = st.sidebar.slider("Number of Recommendations:", min_value=1, max_value=20, value=5)

# Header
st.header(f"Movie Recommendations for User ID: {user_id}")

# Submit Button
if st.button("Get Recommendations"):
    # Call the API
    with st.spinner("Fetching recommendations..."):
        try:
            # Send POST request to the API
            response = requests.post(
                API_URL,
                json={"user_id": user_id, "top_k": top_k}
            )

            # Check if the response is successful
            if response.status_code == 200:
                # Parse recommendations
                recommendations = response.json()

                # Convert recommendations to a Pandas DataFrame
                df = pd.DataFrame(recommendations)
                df.rename(columns={
                    "movie_id": "Movie ID",
                    "name": "Name",
                    "genres": "Genres",
                    "score": "Score"
                }, inplace=True)

                # Display results
                st.success(f"Top {top_k} Recommendations for User ID {user_id}:")
                
                # Display the interactive table in a wider format
                #st.subheader("Interactive Table View")
                st.dataframe(df, use_container_width=True)
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")
