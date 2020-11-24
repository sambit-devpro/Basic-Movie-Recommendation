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

popular = pd.DataFrame()
popular = recm_movies.copy()
popular['popularity'] = recm_movies[recm_movies['popularity'].notnull()]['popularity'].astype('float')
popular = popular.sort_values('popularity',ascending = False)
popular.head()

s = df.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_movies = recm_movies.drop('genres', axis=1).join(s)
gen_movies.head(10)
#gen_movies.columns

df_w = gen_movies[ (gen_movies['genre'] == 'Action') & (gen_movies['vote_count'] >= m)]
df_w.sort_values('Weighted_average', ascending = False).head(10)

df_w = df_w.sort_values('Weighted_average', ascending = False)
plt.figure(figsize=(12,6))
axis1=sns.barplot(x=df_w['Weighted_average'].head(10), y=df_w['title'].head(10), data=df_w)
plt.xlim(4, 10)
plt.title('Best Action Movies by weighted average', weight='bold')
plt.xlabel('Weighted Average Score', weight='bold')
plt.ylabel('Action Movie Title', weight='bold')
