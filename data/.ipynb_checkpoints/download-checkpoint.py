import os
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData, download_url, extract_zip
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
from sklearn.metrics import roc_auc_score
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, row_number, array_contains, split
from pyspark.sql.window import Window




def process_genres_spark(movies_df):
    genre_df = movies_df.select("movieId", "genres")
    genre_df = genre_df.withColumn("genres_array", split(col("genres"), '\|'))
    unique_genres = genre_df.selectExpr("explode(genres_array) as genre").distinct().rdd.flatMap(lambda x: x).collect()
    for genre in unique_genres:
        genre_df = genre_df.withColumn(genre, array_contains(col("genres_array"), genre).cast("int"))
    return genre_df, unique_genres

def map_ids_spark(ratings_df):
    user_mapping = ratings_df.select("userId").distinct().withColumn("user_mappedID", row_number().over(Window.orderBy("userId")) - 1)
    movie_mapping = ratings_df.select("movieId").distinct().withColumn("movie_mappedID", row_number().over(Window.orderBy("movieId")) - 1)
    ratings_df = ratings_df.join(user_mapping, on="userId").join(movie_mapping, on="movieId")
    return ratings_df, user_mapping, movie_mapping

def preprocessing(data_size = 'small'):
    # Set Data Size
    url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip' if data_size == "small" else 'https://files.grouplens.org/datasets/movielens/ml-latest.zip'
    dataset_dir = 'ml-latest-small' if data_size == "small" else 'ml-latest'
    
    # Download Dataset
    if not os.path.exists(dataset_dir):
        extract_zip(download_url(url, '.'), '.')
    
    movies_path, ratings_path = f'{dataset_dir}/movies.csv', f'{dataset_dir}/ratings.csv'
    
    if data_size == "small":
        movies_df = pd.read_csv(movies_path, index_col='movieId')
        ratings_df = pd.read_csv(ratings_path)
    else:
        # Initialize Spark Session
        spark = SparkSession.builder \
            .appName("LargeDatasetProcessing") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        movies_df = spark.read.csv(movies_path, header=True, inferSchema=True)
        ratings_df = spark.read.csv(ratings_path, header=True, inferSchema=True)
        #ratings_df = ratings_df.limit(500000)
        ratings_df = ratings_df
    
    # Process Genres
    if data_size == "small":
        movie_feat = torch.from_numpy(movies_df['genres'].str.get_dummies('|').values).float()
    else:
        genre_df, unique_genres = process_genres_spark(movies_df)
        movie_features_df = genre_df.join(movie_mapping, on="movieId") \
                                    .orderBy("movie_mappedID") \
                                    .select([col(genre) for genre in unique_genres])
        movie_features = movie_features_df.toPandas().values
        movie_feat = torch.from_numpy(movie_features).float()
    
    
    # Map Users and Movies
    if data_size == "small":
        unique_user_id = pd.DataFrame({'userId': ratings_df['userId'].unique(), 'mappedID': pd.RangeIndex(len(ratings_df['userId'].unique()))})
        unique_movie_id = pd.DataFrame({'movieId': ratings_df['movieId'].unique(), 'mappedID': pd.RangeIndex(len(ratings_df['movieId'].unique()))})
    
        ratings_user_tensor = torch.from_numpy(pd.merge(ratings_df[['userId']], unique_user_id, on='userId')['mappedID'].values)
        ratings_movie_tensor = torch.from_numpy(pd.merge(ratings_df[['movieId']], unique_movie_id, on='movieId')['mappedID'].values)
    else:
        ratings_df, user_mapping, movie_mapping = map_ids_spark(ratings_df)
        ratings_user_tensor = torch.from_numpy(np.array(ratings_df.select("user_mappedID").rdd.flatMap(lambda x: x).collect()))
        ratings_movie_tensor = torch.from_numpy(np.array(ratings_df.select("movie_mappedID").rdd.flatMap(lambda x: x).collect()))
      
    # Build HeteroData
    data = HeteroData()
    num_users = len(unique_user_id) if data_size == "small" else user_mapping.count()
    num_movies = len(movies_df) if data_size == "small" else movie_mapping.count()
    
    if data_size == "small":
        data['user'].node_id = torch.arange(num_users)
        data['movie'].node_id = torch.arange(num_movies)
        full_movie_feat = torch.zeros(num_movies, movie_feat.size(1))
        for i, real_id in enumerate(unique_movie_id['movieId'].values):
            idx = movies_df.index.get_loc(real_id) if real_id in movies_df.index else None
            if idx is not None:
                full_movie_feat[i] = movie_feat[idx]
        data['movie'].x = full_movie_feat
        data['user', 'rates', 'movie'].edge_index = torch.stack([ratings_user_tensor, ratings_movie_tensor], dim=0)
        data = T.ToUndirected()(data)
    
        # Split Edges
        transform = T.RandomLinkSplit(num_val=0.1, num_test=0.1, edge_types=('user', 'rates', 'movie'))
    else:
        data['user'].node_id = torch.arange(num_users)
        data['movie'].node_id = torch.arange(num_movies)
        data['movie'].x = movie_feat
        data['user', 'rates', 'movie'].edge_index = torch.stack([ratings_user_tensor, ratings_movie_tensor], dim=0)
        data = ToUndirected()(data)
        transform = RandomLinkSplit(num_val=0.1, num_test=0.1, edge_types=('user', 'rates', 'movie'))
    
    train_data, val_data, test_data = transform(data)
    
    train_pos_edge_index = train_data['user', 'rates', 'movie'].edge_index
    val_pos_edge_index = val_data['user', 'rates', 'movie'].edge_index
    test_pos_edge_index = test_data['user', 'rates', 'movie'].edge_index


    return (num_users, num_movies, movie_feat, train_data, val_data, test_data, train_pos_edge_index, val_pos_edge_index, test_pos_edge_index)
