import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
df = pd.read_csv("netflix_titles.csv")


df['cast']=df['cast'].fillna('No Cast Specified')
df['director']=df['director'].fillna('No Director Specified')
df['country'] = df['country'].fillna('Not Mentioned')

df = df.fillna(0)

import nltk
import re
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
df["title"] = df["title"].apply(clean)

#vll use soup column as feature 2 recommend similar content
count = text.CountVectorizer(stop_words="english")
count_matrix = count.fit_transform(df['listed_in'])
similarity = cosine_similarity(count_matrix, count_matrix)

indices = pd.Series(df.index, index=df['title']).fillna(0)

# Create a function to recommend movies
def get_recommendations_new(title, cosine_sim = similarity):

    title=title.replace(' ','').lower()

    if title not in df['title'].unique():
      print('This Movie / Tv Show is not in our database')
      
    else:
      idx = indices[title]

      sim_scores = list(enumerate(cosine_sim[idx]))

      sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

      sim_scores = sim_scores[1:11]

      movie_indices = [i[0] for i in sim_scores]

      return df['title'].iloc[movie_indices]

# Create the main function
def rec():
        st.title("Netflix Movie Recommendations")
        st.header("Enter your preferences below :")
        title = st.selectbox( "Type or select a movie from the dropdown", df['title'].values)
        results = get_recommendations_new(title)
        st.dataframe(results)
    
        st.header("Enter your Movie / Tv Show Details :")
        show_id = st.text_input('Enter show id')	
        Type = st.text_input('Enter Type (Movie / Tv Show)')
        title = st.text_input('Enter Title name')	
        director = st.text_input('Enter Director Name')	
        cast = st.text_input('Enter Cast names')	
        country = st.text_input('Enter Country name')	
        date_added = st.text_input('Enter Date added (in the format September 25, 2021)')	
        release_year = st.text_input('Enter Year (in the format 2022)')	
        rating = st.text_input('Enter name of rating')	
        duration = st.text_input('Enter Duration (in the format 55 min / 2 Season)')	
        listed_in = st.text_input('Enter in which genere it is listed (in the format TV Shows, TV Dramas, ......)')	
        description = st.text_input('Enter Description of the Movie / Tv Show')

        df.append(pd.Series([show_id, Type, title, director, cast, country, date_added, release_year, rating, duration, listed_in, description],
                             index=df.columns),
                   ignore_index=True)
        ok = st.button('Insert Data')
        if ok:
              st.dataframe(df)

z = df.groupby(['rating']).size().reset_index(name='counts')

fig_pie = px.pie(z, values='counts', names='rating', 
                  title='Distribution of Content Ratings on Netflix',
                  color_discrete_sequence=px.colors.qualitative.Set3)

#top 5 successful directors (add series)
df['director']=df['director'].fillna('No Director Specified')
filtered_directors=pd.DataFrame()
filtered_directors=df['director'].str.split(',',expand=True).stack()
filtered_directors=filtered_directors.to_frame()
filtered_directors.columns=['director']
directors=filtered_directors.groupby(['director']).size().reset_index(name='Total Content')
directors=directors[directors.director !='No Director Specified']
directors=directors.sort_values(by=['Total Content'],ascending=False)
directorsTop5=directors.head()
directorsTop5=directorsTop5.sort_values(by=['Total Content'])

fig_bar = px.bar(directorsTop5,x='director',y='Total Content', color='director', title='Top 5 Directors on Netflix')

#top 5 successful actors
df['cast']=df['cast'].fillna('No Cast Specified')
filtered_cast=pd.DataFrame()
filtered_cast=df['cast'].str.split(',',expand=True).stack()
filtered_cast=filtered_cast.to_frame()
filtered_cast.columns=['Actor']
actors=filtered_cast.groupby(['Actor']).size().reset_index(name='Total Content')
actors=actors[actors.Actor !='No Cast Specified']
actors=actors.sort_values(by=['Total Content'],ascending=False)
actorsTop5=actors.head()
actorsTop5=actorsTop5.sort_values(by=['Total Content'])

fig_bar1 = px.bar(actorsTop5,x='Actor',y='Total Content', title='Top 5 Actors on Netflix', color='Actor')

#Top 10 countries with most releases of TV Shows and Movies
df['country'] = df['country'].fillna('Not Mentioned')
df['country'] = df['country'].apply(lambda x: x.split(",")[0])

c=df['country'].value_counts()
c = c.head(10)

fig_bar2 = px.bar(df, x=c.index, y=c, title='Top 10 countries with most releases of TV Shows and Movies')

#Trend of content produced over the years on Netflix
df1=df[['type','release_year']]
df1=df1.rename(columns={"release_year": "Release Year"})
df2=df1.groupby(['Release Year','type']).size().reset_index(name='Total Content')
df2=df2[df2['Release Year']>=2010]
fig_line = px.line(df2, x="Release Year", y="Total Content", color='type',title='Trend of content produced over the years on Netflix')

def show_visualize():
    st.title("Visualization")
    st.write("Visualize page")

    st.plotly_chart(fig_pie, theme="streamlit", use_conatiner_width=True)
    st.plotly_chart(fig_bar, theme="streamlit", use_conatiner_width=True)
    st.plotly_chart(fig_bar1, theme="streamlit", use_conatiner_width=True)
    st.plotly_chart(fig_bar2, theme="streamlit", use_conatiner_width=True)
    st.plotly_chart(fig_line, theme="streamlit", use_conatiner_width=True)

page = st.sidebar.selectbox("Explore Or Predict", ("Recommendation", "Visualization"))

if page == 'Recommendation':
    rec()
else:
    show_visualize()