import os
import torch
import pandas as pd
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData, download_url, extract_zip
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, row_number, array_contains, split, when
from pyspark.sql.window import Window



'''
from pyspark.sql import SparkSession

    Imports SparkSession, the entry point for Apache Spark SQL operations.
    Used for big data processing and distributed computation.

from pyspark.sql.functions import col, lit, row_number, array_contains, split, when

    Imports various Spark SQL functions used for DataFrame transformations:
        col: References a column in a DataFrame.
        lit: Creates a literal (constant) value in a DataFrame.
        row_number: Assigns row numbers to partitions, useful for ranking.
        array_contains: Checks if an array column contains a specific value.
        split: Splits a string column into an array based on a delimiter.
        when: Used for conditional expressions (similar to SQL CASE statements).

from pyspark.sql.window import Window

    Imports Window, which is used for performing window functions (like ranking and aggregations over partitions of data) in Spark DataFrames.


'''


#Missing Titles → Replaced with "Unknown Title".
#Missing Genres → Replaced with "Unknown".
#Missing Ratings → Dropped rows where rating is missing.

def process_genres_spark(movies_df):
    genre_df = movies_df.select("movieId", "genres")
    genre_df = genre_df.withColumn("genres", when(col("genres").isNull(), "Unknown").otherwise(col("genres")))
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

def preprocessing(data_size='small'):
    url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip' if data_size == "small" else 'https://files.grouplens.org/datasets/movielens/ml-latest.zip'
    dataset_dir = 'ml-latest-small' if data_size == "small" else 'ml-latest'
    
    if not os.path.exists(dataset_dir):
        extract_zip(download_url(url, '.'), '.')
    
    movies_path, ratings_path = f'{dataset_dir}/movies.csv', f'{dataset_dir}/ratings.csv'
    
    if data_size == "small":
        movies_df = pd.read_csv(movies_path, index_col='movieId')
        ratings_df = pd.read_csv(ratings_path)
        
        # Handle missing values
        movies_df.loc[:, 'title'] = movies_df['title'].fillna("Unknown Title")
        movies_df.loc[:, 'genres'] = movies_df['genres'].fillna("Unknown")
        ratings_df = ratings_df.dropna(subset=['rating'])
        
        movie_id_to_name = movies_df['title'].to_dict()
        genre_names = movies_df['genres'].str.get_dummies('|').columns.tolist()
        movie_feat = torch.from_numpy(movies_df['genres'].str.get_dummies('|').values).float()
        
        unique_user_id = pd.DataFrame({'userId': ratings_df['userId'].unique(), 'mappedID': pd.RangeIndex(len(ratings_df['userId'].unique()))})
        unique_movie_id = pd.DataFrame({'movieId': ratings_df['movieId'].unique(), 'mappedID': pd.RangeIndex(len(ratings_df['movieId'].unique()))})
        
        ratings_user_tensor = torch.from_numpy(pd.merge(ratings_df[['userId']], unique_user_id, on='userId')['mappedID'].values)
        ratings_movie_tensor = torch.from_numpy(pd.merge(ratings_df[['movieId']], unique_movie_id, on='movieId')['mappedID'].values)
        
    else:
        spark = SparkSession.builder.appName("LargeDatasetProcessing").config("spark.executor.memory", "4g").config("spark.driver.memory", "4g").getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        
        movies_df = spark.read.csv(movies_path, header=True, inferSchema=True)
        ratings_df = spark.read.csv(ratings_path, header=True, inferSchema=True)
        
        # Handle missing values in Spark DataFrames
        movies_df = movies_df.withColumn("title", when(col("title").isNull(), "Unknown Title").otherwise(col("title")))
        movies_df = movies_df.withColumn("genres", when(col("genres").isNull(), "Unknown").otherwise(col("genres")))
        ratings_df = ratings_df.na.drop(subset=["rating"])
        
        ratings_df, user_mapping, movie_mapping = map_ids_spark(ratings_df)
        genre_df, unique_genres = process_genres_spark(movies_df)
        
        movie_features_df = genre_df.join(movie_mapping, on="movieId").orderBy("movie_mappedID").select([col(genre) for genre in unique_genres])
        movie_features = movie_features_df.toPandas().values
        movie_feat = torch.from_numpy(movie_features).float()
        genre_names = unique_genres
        movie_id_to_name = movies_df.select("movieId", "title").toPandas().set_index("movieId")["title"].to_dict()
        
        ratings_user_tensor = torch.from_numpy(np.array(ratings_df.select("user_mappedID").rdd.flatMap(lambda x: x).collect()))
        ratings_movie_tensor = torch.from_numpy(np.array(ratings_df.select("movie_mappedID").rdd.flatMap(lambda x: x).collect()))
    
    data = HeteroData()
    num_users = len(unique_user_id) if data_size == "small" else user_mapping.count()
    num_movies = len(movies_df) if data_size == "small" else movie_mapping.count()
    
    data['user'].node_id = torch.arange(num_users)
    data['movie'].node_id = torch.arange(num_movies)
    
    if data_size == "small":
        full_movie_feat = torch.zeros(num_movies, movie_feat.size(1))
        for i, real_id in enumerate(unique_movie_id['movieId'].values):
            idx = movies_df.index.get_loc(real_id) if real_id in movies_df.index else None
            if idx is not None:
                full_movie_feat[i] = movie_feat[idx]
        data['movie'].x = full_movie_feat
    else:
        data['movie'].x = movie_feat
    
    data['user', 'rates', 'movie'].edge_index = torch.stack([ratings_user_tensor, ratings_movie_tensor], dim=0)
    data = T.ToUndirected()(data)
    
    transform = T.RandomLinkSplit(num_val=0.1, num_test=0.1, edge_types=('user', 'rates', 'movie'))
    train_data, val_data, test_data = transform(data)
    
    train_pos_edge_index = train_data['user', 'rates', 'movie'].edge_index
    val_pos_edge_index = val_data['user', 'rates', 'movie'].edge_index
    test_pos_edge_index = test_data['user', 'rates', 'movie'].edge_index
    
    return num_users, num_movies, movie_feat, train_data, val_data, test_data, train_pos_edge_index, val_pos_edge_index, test_pos_edge_index, genre_names, movie_id_to_name
