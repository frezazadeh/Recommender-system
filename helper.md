# MovieLens Dataset Analysis

## 1. Data Overview

We are working with the MovieLens dataset, a widely used dataset in recommendation systems. It contains:

- **610 users**
- **9,724 movies**
- **100,836 ratings**
- **Average ratings per user:** 165.30
- **Average ratings per movie:** 10.37
- **Sparsity of the rating matrix:** 98.3%, meaning most movies do not have ratings from most users.

---

## 2. Rating Distribution

### Key Observations:

- The ratings are not uniformly distributed.
- Most users tend to give higher ratings, with **4.0 being the most common rating (26.6%)**, followed by **3.0 (19.88%)**.
- There are fewer low ratings (0.5, 1.0, 1.5), indicating that users are generally positive in their ratings.

### 📊 Plot Analysis (First Plot):

- The bar chart visualizes the frequency of each rating.
- The skew towards higher ratings suggests that users tend to rate movies positively.

### Potential Implications for Model:

- A recommendation system might over-recommend highly rated movies unless balanced with user preferences.
- Since the dataset lacks many negative ratings, a **binary classification approach (liked/disliked)** could be an alternative modeling approach.

---

## 3. Ratings per User

### Key Observations:

- The number of ratings per user follows a highly skewed distribution.
- Some users have rated only **20 movies**, while others have rated **over 2,600 movies**.
- The **median number of ratings per user is 70.5**, meaning half of the users have rated fewer than 71 movies.

### 📊 Plot Analysis (Second Plot):

- A long-tail distribution with a few users contributing most ratings.
- Most users rate fewer than 100 movies.
- The smooth curve (KDE) highlights the drop-off in rating activity.

### Potential Implications for Model:

- **Cold Start Problem:** Users with very few ratings make personalized recommendations difficult.
- A **user-based collaborative filtering approach** might struggle due to the uneven distribution of ratings.

---

## 4. Ratings per Movie

### Key Observations:

- Some movies are extremely popular, with over **300 ratings**.
- The majority of movies have very few ratings (**many with just 1 or 2 ratings**).
- The **median number of ratings per movie is 3**, meaning most movies are rarely rated.

### 📊 Plot Analysis (Third Plot):

- Similar to user ratings, movies follow a **long-tail distribution**.
- Popular movies like **The Matrix (1999), Star Wars (1977), and Shawshank Redemption (1994)** have the most ratings.
- Many movies have limited engagement, making it harder to recommend lesser-known ones.

### Potential Implications for Model:

- **Popular Bias:** Recommendation models might favor frequently rated movies over lesser-known ones.
- **Hybrid Models** combining content-based filtering with collaborative filtering could help recommend lesser-known movies based on their features.

---

## 5. Most & Least Rated Movies

### 📌 Top 10 Most Rated Movies:

- Classic, popular films like **The Matrix, Star Wars, and Forrest Gump** appear frequently.
- These movies tend to have **Action, Drama, and Sci-Fi** genres, indicating a user preference trend.

### 📌 Top 10 Least Rated Movies:

- Movies with very few ratings are mostly **obscure, niche, or foreign films**.
- **Comedy and Drama** genres appear frequently among the least rated movies.

### Potential Implications for Model:

- **Cold Start Problem for Movies:** Many movies have too few ratings to make reliable predictions.
- **Using a hybrid recommendation model (collaborative + content-based filtering)** can improve recommendations.

---

## 6. Key Challenges Identified

🔹 **Sparse Data:** 98.3% of the user-item matrix is empty. **Matrix factorization (SVD, ALS)** or **deep learning techniques (Neural Collaborative Filtering)** could improve recommendations.

🔹 **Cold Start Problem:** New users and movies with very few ratings make it hard to generate meaningful recommendations.

🔹 **Bias Towards Popular Movies:** The model might over-recommend popular movies unless mitigated by diversity-promoting techniques.

🔹 **Skewed Rating Distribution:** Many ratings are clustered around **3.0-5.0**, which could make it harder to differentiate good recommendations from bad ones.

---

## 7. Next Steps

📌 **Feature Engineering:** Extract movie metadata (**genre, year, actors**) for content-based recommendations.

📌 **Model Selection:** Consider **Collaborative Filtering (Matrix Factorization, Neural Networks)** combined with **Content-Based Filtering** for better recommendations.

📌 **Evaluation Metrics:** Use **RMSE, MAE, and Precision@K** to validate model effectiveness.

📌 **Handling Data Sparsity:** Implementing **item-based collaborative filtering** or **embedding-based deep learning models** could improve recommendations.

---


# Movie Recommendation System: GNN and BPR Approach

## Introduction
The goal of this project is to build a movie recommendation system using machine learning techniques. The MovieLens dataset is used, which contains user ratings for movies. The model aims to predict user preferences and recommend movies they are likely to enjoy based on their past interactions.

This document explains why Graph Neural Networks (GNNs) and Bayesian Personalized Ranking (BPR) were chosen for the recommendation model.

---

## Why Use Graph Neural Networks (GNNs)?

### 1. **Graph Structure in Movie Recommendations**
The MovieLens dataset can naturally be represented as a bipartite graph where:
- **Users** and **movies** are the nodes.
- **Edges** represent interactions (e.g., ratings or views).
- Additional features such as movie genres provide side information.

GNNs effectively capture the relationships between users and items by learning meaningful embeddings from this graph structure.

### 2. **Message Passing and Representation Learning**
GNNs use message passing to propagate information across connected nodes. In this model:
- A user's embedding is influenced by the embeddings of movies they have interacted with.
- A movie's embedding is influenced by users who have interacted with it.
- This allows for a **better generalization** of user preferences even when explicit ratings are sparse.

### 3. **Handling Sparsity and Cold Start Problems**
Traditional recommendation systems (e.g., collaborative filtering) struggle with:
- **Sparse interactions** (not every user has rated many movies).
- **Cold start issues** (new users or new movies have little to no interaction data).

GNNs leverage **inductive learning**, where the model generalizes from existing relationships and attributes, improving recommendations for new users or movies.

### 4. **Scalability with Heterogeneous Graphs**
GNNs, especially the **GraphSAGE (Sample and Aggregate) variant used here**, scale efficiently for large datasets by:
- Sampling neighbors instead of considering the entire graph.
- Aggregating information to form compact and effective representations.

This makes it suitable for handling larger datasets like **MovieLens-Latest**.

---

## Why Use Bayesian Personalized Ranking (BPR)?

### 1. **Implicit Feedback Modeling**
BPR is designed for **implicit feedback** settings, where explicit ratings are unavailable or unreliable. Instead of predicting exact ratings, BPR optimizes the model to rank positive interactions higher than negative ones.

### 2. **Pairwise Ranking Loss**
BPR optimizes the objective:
\[
\sum_{(u, i, j) \in D} \log(\sigma(x_{ui} - x_{uj}))
\]
where:
- \( u \) is a user.
- \( i \) is a positively interacted movie.
- \( j \) is a randomly sampled negative movie.
- \( x_{ui} \) and \( x_{uj} \) are scores given by the model.
- The loss ensures that a user prefers movie \( i \) over \( j \) with a higher probability.

### 3. **Optimizing Ranking Instead of Prediction**
Unlike mean squared error (MSE), which predicts exact ratings, BPR directly optimizes ranking quality, making it more effective for recommendation tasks where relative preferences matter more than absolute ratings.

### 4. **Improved Personalization**
BPR adapts to each user's implicit preferences, making it ideal for:
- **Personalized recommendations** where ranking matters.
- **Sparse datasets** where explicit user feedback is minimal.

---

## How the Model Works

1. **Preprocessing**
   - The MovieLens dataset is processed into a graph structure with users and movies as nodes and interactions as edges.
   - Features like genres are included for better representations.

2. **Graph Neural Network (GNN) Training**
   - A **GraphSAGE-based GNN** is trained to learn node embeddings.
   - The embeddings capture user preferences and movie properties.

3. **BPR Training**
   - The learned embeddings are used in a **Bayesian Personalized Ranking (BPR) loss** function.
   - The model optimizes ranking by ensuring positive interactions rank higher than negative ones.

4. **Inference (Recommendation Generation)**
   - A user's embedding is matched with movie embeddings.
   - The model scores and ranks movies for recommendation.
   - The top \( k \) movies are selected for the user.

---

## Conclusion
The combination of **GNN and BPR** is well-suited for the movie recommendation problem due to:
- **Graph-based learning** to capture complex user-movie relationships.
- **Message passing for feature aggregation**, improving generalization.
- **Pairwise ranking optimization** for better personalized recommendations.
- **Scalability for large datasets** using GraphSAGE and mini-batch training.

This approach improves recommendation accuracy and adaptability compared to traditional collaborative filtering methods.

---

# **Discussion: Addressing Key Considerations in Movie Recommendation System**

## **Cold-Start Problem**

### **Issue:**
- New users have **limited or no interaction history**, making personalized recommendations difficult.
- New movies with few ratings may not be recommended effectively.

### **Solution Implemented:**
1. **Graph-Based Approach (GNN):**
   - Even if a new user has no past ratings, they can still receive recommendations based on movie features (**content-based filtering**).
   - The GNN model learns relationships between users and movies through indirect connections in the graph.

2. **Content-Based Features:**
   - Movie **genres** are incorporated into embeddings, ensuring **cold-start movies** can be recommended based on their genre similarity to well-rated movies.

3. **Hybrid Recommendation System:**
   - Collaborative filtering (GNN) + Content-based filtering (Movie Features) ensures that users with few ratings still get meaningful recommendations.
   - Future improvements: **Integrate user profile metadata** (e.g., age, location) to enhance recommendations.

---

## **Scalability**

### **Issue:**
- Handling **large datasets** efficiently and ensuring **real-time recommendation** capabilities.
- MovieLens dataset is relatively small, but in real-world applications, the system should scale to millions of users and items.

### **Solution Implemented:**
1. **Apache Spark for Large Data Processing:**
   - Uses Spark **for preprocessing large datasets**, efficiently handling mapping of user/movie IDs and feature extraction.

2. **Efficient Graph Representation:**
   - **PyTorch Geometric (PyG)** enables handling large-scale graph data efficiently.
   - Sparse adjacency matrices ensure efficient memory usage.

3. **Fast Inference & API Optimization:**
   - **FastAPI** backend ensures quick response times for real-time recommendations.
   - Model is optimized to run on **CUDA (GPU acceleration)** for scalable inference.
   - **Batch processing & caching strategies** could be added to improve response time further.

4. **Deployment-Ready Architecture:**
   - Model can be **deployed as a microservice** (Docker + Kubernetes) for horizontal scaling.
   - Load balancing techniques can be incorporated for handling high user traffic.

---

## **User Feedback Integration**

### **Issue:**
- Static models degrade over time as user preferences change.
- Need a way to **incorporate new interactions** to improve recommendations dynamically.

### **Solution Implemented:**
1. **Continuous Model Training:**
   - A pipeline can be implemented where **new user interactions (ratings, clicks, skips)** update the model periodically.
   - **Online learning** or **incremental retraining** on new data batches can improve model adaptation.

2. **Implicit Feedback Learning:**
   - Not just explicit ratings—**watch time, re-watches, skips** can be integrated into training.
   - Bayesian Personalized Ranking (BPR) already considers implicit feedback by ranking preferences.

3. **User Feedback Loop via API & UI:**
   - Allow users to **rate recommendations** directly through the Streamlit UI.
   - Use feedback to fine-tune recommendations (**reward good predictions, penalize bad ones**).

---

## **Ethical Considerations & Bias Mitigation**

### **Issue:**
- **Bias in Recommendations:**
  - Popular movies dominate recommendations (**popularity bias**), making it harder for lesser-known movies to be recommended.
  - Genre, age, or gender biases in ratings could reinforce societal stereotypes.
  
- **Fairness & Diversity:**
  - Users should not be limited to only certain types of content (e.g., recommending only action movies if they once watched an action film).

### **Solution Implemented:**
1. **Debiasing the Model:**
   - **Regularization in BPR Loss** to prevent overfitting on popular movies.
   - Penalizing repeated recommendations of **only highly rated** or **popular movies**.

2. **Diverse & Serendipitous Recommendations:**
   - Implementing **diversity-promoting strategies**, such as:
     - **Exploration-Exploitation Tradeoff:** Introduce **lesser-known movies** along with popular ones.
     - **Re-ranking Techniques:** Ensure genre and popularity balance in recommendations.
   
3. **Transparency & User Control:**
   - Users should have the option to **customize recommendations** (e.g., explore new genres).
   - Providing explanations on **why a movie was recommended** (e.g., "Recommended because you watched Inception").

---

## **Future Enhancements**
- **Better Cold-Start Handling**:
  - Incorporate **user demographics & behavior-based recommendations**.
- **Scalability Improvements**:
  - Explore **graph partitioning & distributed GNN training**.
- **Bias Reduction & Fairness**:
  - Implement **adversarial training** to mitigate biases in movie recommendations.
- **Improved Real-Time Learning**:
  - Implement **reinforcement learning** to dynamically adjust recommendations based on real-time feedback.

---

## **Conclusion**
The current implementation successfully addresses major challenges in movie recommendations, leveraging **GNNs for learning user-movie interactions** and **BPR for ranking optimization**. Future improvements will focus on **real-time adaptation, bias reduction, and user-driven customization** to make recommendations more personalized and ethical.

---




### 📢 Stay Tuned for More Insights!

This analysis sets the foundation for building an effective movie recommendation system. Further experimentation and fine-tuning will enhance performance and improve user experience.

---

