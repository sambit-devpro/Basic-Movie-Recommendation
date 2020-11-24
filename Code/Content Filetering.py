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

cont_recm = recm_movies.copy()
cont_recm.head()
from sklearn.feature_extraction.text import TfidfVectorizer
tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), stop_words = 'english')
cont_recm['overview'] = cont_recm['overview'].fillna('')
# Fitting the TF-IDF on the 'overview' text
tfv_matrix = tfv.fit_transform(cont_recm['overview'])
#Finding Cosine_similarity
cos_sim = linear_kernel(tfv_matrix, tfv_matrix)
cont_recm = cont_recm.reset_index()
indices = pd.Series(cont_recm.index, index=cont_recm['title'])
indices.head(20)
def sugg_recm(title):
    # Get the index corresponding to original_title
    idx = indices[title]    
    # Get the pairwsie similarity scores 
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies 
    sim_scores = sorted(sim_scores, key=lambda x: x[1],reverse=True)
    
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]
    
    sugg_recm('Star Wars').head(10)
