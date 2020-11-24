import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ast 
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel,cosine_similarity
%matplotlib inline

movies_df = pd.read_csv('movies_metadata.csv')
ratings = pd.read_csv('ratings_small.csv')
movies_df.columns

movies_df = movies_df.drop(['belongs_to_collection', 'budget', 'homepage', 'original_language', 'release_date', 'revenue', 'runtime', 'spoken_languages', 'status', 'video', 'poster_path', 'production_companies', 'production_countries'], axis = 1)
movies_df.head()


movies_df['genres'] = movies_df['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
movies_df['genres'].head()

#vote_count
V = movies_df[movies_df['vote_count'].notnull()]['vote_count'].astype('float')
#vote_average
R = movies_df[movies_df['vote_average'].notnull()]['vote_average'].astype('float')
#vote_average_mean
C = R.mean()
#minimum votes required to get in list
M = V.quantile(0.95)

df = pd.DataFrame()
df = movies_df[(md['vote_count'] >= m) & (movies_df['vote_average'].notnull())][['title','vote_count','vote_average','popularity','genres','overview']]

df['Weighted_average'] = ((R*V) + (C*M))/(V+M)
recm_movies = df.sort_values('Weighted_average', ascending=False).head(500)
recm_movies.head()
