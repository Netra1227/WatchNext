import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots


import matplotlib.pyplot as plt

import seaborn as sns

data = pd.read_csv("netflix_titles.csv")
date_replacements = {
    2288: 'January 1, 2015'
}

for id, date in date_replacements.items():
    data.iloc[id,4] = date

data = data[data['date_added'].notna()]
#Country
data['country'] = data['country'].fillna(data['country'].mode()[0])

#Remove Unwanted Data
data = data.drop(['director', 'cast'], axis=1)

data['year_added'] = data['date_added'].apply(lambda x: x.split(" ")[-1])
data['month_added'] = data['date_added'].apply(lambda x: x.split(" ")[0])

#Targets Based on rating
ratings_ages = {
    'TV-PG': 'Older Kids',
    'TV-MA': 'Adults',
    'TV-Y7-FV': 'Older Kids',
    'TV-Y7': 'Older Kids',
    'TV-14': 'Teens',
    'R': 'Adults',
    'TV-Y': 'Kids',
    'NR': 'Adults',
    'PG-13': 'Teens',
    'TV-G': 'Kids',
    'PG': 'Older Kids',
    'G': 'Kids',
    'UR': 'Adults',
    'NC-17': 'Adults'
}

data['age_level'] = data['rating'].replace(ratings_ages)
#Multiple Countries to One
data['principal_country'] = data['country'].apply(lambda x: x.split(",")[0])
# type should be a category
data['type'] = pd.Categorical(data['type'])
# target_ages is another category (4 classes)
data['age_level'] = pd.Categorical(data['age_level'], categories=['Kids', 'Older Kids', 'Teens', 'Adults'])

# Year added should be integer so we can compare with `released_year`
data['year_added'] = pd.to_numeric(data['year_added'])

#Genres 
data['genre'] = data['listed_in'].apply(lambda x :  x.replace(' ,',',').replace(', ',',').split(',')) 


movie = data[data['type'] == 'Movie']
show = data[data['type'] == 'TV Show']
def generate_rating_df(data):
    rating = data.groupby(['rating', 'age_level']).agg({'show_id': 'count'}).reset_index()
    rating = rating[rating['show_id'] != 0]
    rating.columns = ['rating', 'age_level', 'counts']
    rating = rating.sort_values('age_level')
    return rating

rating = generate_rating_df(data)
gen_fig = px.histogram(rating, x='rating', y='counts', color='age_level')

movie_rating = generate_rating_df(movie)
show_rating = generate_rating_df(show)

mt_fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]])

mt_fig.add_trace(
    go.Pie(labels=movie_rating['age_level'], values=movie_rating['counts']),
    row=1, col=1
)

mt_fig.add_trace(
    go.Pie(labels=show_rating['age_level'], values=show_rating['counts']),
    row=1, col=2
)

mt_fig.update_traces(textposition='inside', hole=.4, hoverinfo="label+percent+name")
mt_fig.update_layout(annotations=[dict(text='Movies', x=0.20, y=0.5, font_size=14, showarrow=False),
                 dict(text='TV Shows', x=0.80, y=0.5, font_size=14, showarrow=False)])


#Movie and TV Show Released and Added on Netflix Line Graph
released_year_df = data.loc[data['release_year'] > 2010].groupby(['release_year', 'type']).agg({'show_id': 'count'}).reset_index()
added_year_df = data.loc[data['year_added'] > 2010].groupby(['year_added', 'type']).agg({'show_id': 'count'}).reset_index()

lay = go.Layout(
    xaxis=dict(
        title="Year"
    ),
    yaxis=dict(
        title="Movies/Shows Added on Netflix"
    ) ) 

yfig = go.Figure(layout=lay)
yfig.add_trace(go.Scatter( 
    x=released_year_df.loc[released_year_df['type'] == 'Movie']['release_year'], 
    y=released_year_df.loc[released_year_df['type'] == 'Movie']['show_id'],
    mode='lines+markers',
    name='Movie: Released Year',
    marker_color='red',
))
yfig.add_trace(go.Scatter( 
    x=released_year_df.loc[released_year_df['type'] == 'TV Show']['release_year'], 
    y=released_year_df.loc[released_year_df['type'] == 'TV Show']['show_id'],
    mode='lines+markers',
    name='TV Show: Released Year',
    marker_color='blue',
))
yfig.add_trace(go.Scatter( 
    x=added_year_df.loc[added_year_df['type'] == 'Movie']['year_added'], 
    y=added_year_df.loc[added_year_df['type'] == 'Movie']['show_id'],
    mode='lines+markers',
    name='Movie: Year Added',
    marker_color='green',
))
yfig.add_trace(go.Scatter( 
    x=added_year_df.loc[added_year_df['type'] == 'TV Show']['year_added'], 
    y=added_year_df.loc[added_year_df['type'] == 'TV Show']['show_id'],
    mode='lines+markers',
    name='TV Show: Year Added',
    marker_color='black',
))
yfig.update_xaxes(categoryorder='total descending')

#Genre Classfication

from sklearn.preprocessing import MultiLabelBinarizer
def calculate_mlb(series):
    mlb = MultiLabelBinarizer()
    mlb_data = pd.DataFrame(mlb.fit_transform(series), columns=mlb.classes_, index=series.index)
    return mlb_data

def top_genres(df, title='Top ones'):
    genres_df = calculate_mlb(data['genre'])
    tdata = genres_df.sum().sort_values(ascending=False)
    return tdata


tdata = top_genres(movie,title='Top Movie Genre')
lay2 = go.Layout(
    xaxis=dict(
        title="Genres"
    ),
    yaxis=dict(
        title="Data Count"
    ) ) 
mfig = go.Figure(layout = lay2)
mfig.add_trace(go.Bar(
    x=tdata.index,
    y=tdata.values,
    ))
mfig.update_xaxes(title_standoff = 25,tickangle = 30)

country = data['principal_country'].value_counts().reset_index()
country = country[country['principal_country'] /  country['principal_country'].sum() > 0.01]


merged = pd.read_csv("merged.csv")  
print(merged.head(5))

merged[merged['country'].isnull()]
merged.loc[6, 'country'] = 'India'
merged.loc[74, 'country'] = 'United Kingdom'
merged.loc[260, 'country'] = 'India'
merged.loc[506, 'country'] = 'Indonesia'

from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2

for i in range(len(merged['country'])):
    split = merged['country'].iloc[i].split(',')
    merged['country'].iloc[i] = split[0]
    if merged['country'].iloc[i] == 'Soviet Union':
        merged['country'].iloc[i] = 'Russia'

continents = {
    'AF': 'Africa',
    'AS': 'Asia',
    'OC': 'Australia',
    'EU': 'Europe',
    'NA': 'North America',
    'SA': 'South America'
}
countries = merged['country']

merged['continent'] = [continents[country_alpha2_to_continent_code(country_name_to_country_alpha2(country))] for country in countries]
print(merged.head(5))
print(merged.index)



app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

fig1 = px.pie(data['type'].value_counts().reset_index(), values='type', names='index')
fig1.update_traces(textposition='inside', textinfo='percent+label')

graph = html.Div([
    html.Div([
        html.Div([
            dcc.Graph(id='g1', figure=fig1)
        ], className="six columns",style={'height':'395px'}),
])])

card2 = dbc.Card(
    [
    dbc.CardImg(src="https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/summer-movies-1587392939.jpg", top=True),
    dbc.CardBody(
        [
            html.Center(html.H5("Movies", className="card-title")),
            html.Center(html.P(len(movie))),
            html.Center(html.P("Total Number of Movies present in Dataset")),
        ]
    )
    ],
    style = {'height':'367px'}
)
card3 = dbc.Card(
    [
        dbc.CardImg(src="https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/teen-shows-1582317535.jpg", top=True),
    dbc.CardBody(
        [
            html.Center(html.H5("TV Shows", className="card-title")),
            html.Center(html.P(len(show))),
            html.Center(html.P("Total Number of Shows present in Dataset")),
        ]
    )
        ],
    style = {'height':'363px'}
)

cardlast = dbc.Card(
    [
        dbc.CardImg(src="https://deadline.com/wp-content/uploads/2020/07/netflix-logo.png", top=True),
    dbc.CardBody(
        [
            html.Center(html.H5("Netflix Data Analysis Dashboard and Recommendation System", className="card-title")),
            html.Center(html.P("Netravati P")),
        ]
    )
        ],
    style = {'height':'367px','border-style': 'none','margin-top':'98px'}
)


navbar = dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                    dbc.Col(dbc.NavbarBrand("Netflix Dashboard", className="ml-2")),
                ],
                align="center",
                no_gutters=True,
            ),
            href="https://plot.ly",
        ),
    ],
    color="dark",
    dark=True,
)

opts = [
        {'label': 'Africa', 'value': 'AF'},
        {'label': 'Asia', 'value': 'AS'},
        {'label': 'Australia', 'value': 'OC'},
        {'label': 'Europe', 'value': 'EU'},
        {'label': 'North America', 'value': 'NA'},
        {'label': 'South America', 'value': 'SA'}
        ]

optsage = [
    {'label':'weighted_average_vote','value':'Average'},
    {'label':'weighted_allgenders_0age','value':'0age'},
    {'label':'weighted_allgenders_18age','value':'18age'},
    {'label':'weighted_allgenders_30age','value':'30age'},
    {'label':'weighted_allgenders_45age','value':'45age'}
]

optsgen = [
    {'label':'Dramas','value':'DR'},
    {'label':'International Movies','value':'IN'},
    {'label':'Action & Adventure','value':'AD'},
    {'label':'Comedies','value':'CO'},
    {'label':'Classic Movies','value':'CL'},
    {'label':'Thrillers','value':'TH'},
    {'label':'Independent Movies','value':'IM'},
    {'label':'Sci-Fi & Fantasy','value':'SF'},
    {'label':'Cult Movies','value':'CM'},
    {'label':'Sports Movies','value':'SM'},
    {'label':'Romantic Movies','value':'RM'},
    {'label':'Children & Family Movies','value':'CFM'},
    {'label':'LGBTQ Movies','value':'LGB'},
    {'label':'Horror Movies','value':'HM'},
    {'label':'Stand-Up Comedy','value':'SUC'}
    ]

optsmf = [
    {'label':'Male','value':'M'},
    {'label':'Female','value':'F'}
    ]

optsrec = [
    {'label':'Transformers: Robots in Disguise','value':'Transformers'},
    {'label':'Inception','value':'Inception'},
	{'label':'3 Idiots','value':'3 Idiots'},
    {'label':'Gol Maal','value':'Gol Maal'},
    {'label':'3 Days to Kill','value':'3 Days to Kill'},
    {'label':'Race 2','value':'Race 2'},
    {'label':'Rango','value':'Rango'},
    {'label':'Rock On!','value':'Rock On!'}

    ]
 


def top10gender(column):
    titles = []
    scores = []
    top10=merged[column].nlargest(10)
    for i in range(len(top10)):
        index = top10.index[i]
        score = top10.iloc[i]
        scores.append(score)
        title = merged['title'].iloc[index]
        titles.append(title)
        print(i+1, '.', title, ':', score)
    topmf = pd.DataFrame(list(zip(titles, scores)),columns =['Titles', 'Scores'])
    return topmf

def top10byage(column):
    titles = []
    scores = []
    top10=merged[column].nlargest(10)
    for i in range(len(top10)):
        index = top10.index[i]
        score = top10.iloc[i]
        scores.append(score)
        title = merged['title'].iloc[index]
        titles.append(title)
        print(i+1, '.', title, ':', score)
    topa = pd.DataFrame(list(zip(titles, scores)),columns =['Titles', 'Scores'])
    return topa

def continentstop10(continent_string, vote):
    titles = []
    scores = []
    continent = []
    for i in range(len(merged)):
        row = merged.iloc[i]
        if row['continent'] == continent_string:
            continent.append(row)
    continent = pd.DataFrame(continent)
    top10=continent[vote].nlargest(10)
    for i in range(len(top10)):
        index = top10.index[i]
        score = top10.iloc[i]
        scores.append(score)
        title = merged['title'].iloc[index]
        titles.append(title)
        print(i+1, '.', title, ':', score)
    topm = pd.DataFrame(list(zip(titles, scores)),columns =['Titles', 'Scores'])
    return topm

def genrestop10(genre_string, column):
    titles = []
    scores = []
    genre = []
    for i in range(len(merged)):
        row = merged.iloc[i]
        if genre_string in row['listed_in']:
            genre.append(row)
    genre = pd.DataFrame(genre)
    top10=genre[column].nlargest(10)
    for i in range(len(top10)):
        index = top10.index[i]
        score = top10.iloc[i]
        scores.append(score)
        title = merged['title'].iloc[index]
        titles.append(title)
        print(i+1, '.', title, ':', score)
    topg = pd.DataFrame(list(zip(titles, scores)),columns =['Titles', 'Scores'])
    return topg

import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
# Capture similarity 
import dash_table
from sklearn.metrics.pairwise import linear_kernel

datarec = pd.read_csv("netflix_titles.csv")
datarec.drop(["show_id","director","cast","country","date_added","release_year","rating","duration"],axis=1,inplace=True)
datarec.head(10)
datarec['listed_in'] = [re.sub(r'[^\w\s]', '', t) for t in datarec['listed_in']]
datarec['description'] = [re.sub(r'[^\w\s]', '', t) for t in datarec['description']]

datarec['listed_in'] = [t.lower() for t in datarec['listed_in']]
datarec['description'] = [t.lower() for t in datarec['description']]
datarec["combined"] = datarec['listed_in'] + '  ' + datarec['title'] + ' ' + datarec['description'] 
data.drop(["description","listed_in","type"],axis=1,inplace=True)
# Content Similarity
vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(datarec["combined"])
cosine_similarities = linear_kernel(matrix,matrix)
movie_title = datarec['title']
indices = pd.Series(datarec.index, index=datarec['title'])

def content_recommender(title):
	idx = indices[title]
	sim_scores = list(enumerate(cosine_similarities[idx]))
	sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
	sim_scores = sim_scores[1:31]
	movie_indices = [i[0] for i in sim_scores]
	cm = pd.DataFrame(movie_title.iloc[movie_indices])
	return cm.head(10)

reco = content_recommender('Transformers: Robots in Disguise')
print(reco)

app.layout = dbc.Container(fluid=True, style={'backgroundColor':'#ffe6e6'}, children=[
    ## Top
    html.H1(id="nav-pills"),
    navbar,
    html.Br(),
    
    dbc.Row([
        dbc.Col(md=6, children=[
            graph,
            html.Br(),
            ]), 
        dbc.Col(md=3, children=[
            card2,
            html.Br(),
            ]),
        dbc.Col(md=3, children=[
            card3,
            html.Br(),
            ])
        ]),
    html.Br(),html.Br(),
    dbc.Row([
        dbc.Col(md=6, children=[
            html.Div(children=[
                html.H1(children='''
                        Top Movies in Each Continent'''),
                html.Br(),
                dcc.Dropdown(
                    id = "input",
                    options=optsage,
                    value=optsage[0]
                    ),
                html.Div([
                    html.Div([
                        html.Div([
                            dcc.Graph(id='graph-court')
                            ], className="six columns",style={'height':'495px'}),
                ])
                ])
                ])
            ]),
        
        dbc.Col(md=6, children=[
            html.Div(children=[
                html.H1(children='''Top Movies in By Age Group'''),
                html.Br(),
                dcc.Dropdown(
                    id = "inputage",
                    options=optsage,
                    value=optsage[0]
                    ),
                html.Div([
                    html.Div([
                        html.Div([
                            dcc.Graph(id='graph-court2')
                            ], className="six columns",style={'height':'395px'}),
                        ])
                    ])
                ])
            ])
        ]),
    html.Br(),html.Br(),
    dbc.Row([
        dbc.Col(md=6, children=[
            html.Div(children=[
                html.H1(children='''
                        Top Movies by Genres'''),
                html.Br(),
                dcc.Dropdown(
                    id = "inputgenre",
                    options=optsgen,
                    value=optsgen[0]),
                html.Div([
                    html.Div([
                        html.Div([
                            dcc.Graph(id='graph-court3')
                            ], className="six columns",style={'height':'395px'}),
                        ])
                    ])
                ])
            ]),
        dbc.Col(md=6, children=[
            html.Div(children=[
                html.H1(children='''Top Movies in By Gender'''),
                html.Br(),
                dcc.Dropdown(
                    id = "input4",
                    options=optsmf,
                    value=optsmf[0]
                    ),
                html.Div([
                    html.Div([
                        html.Div([
                            dcc.Graph(id='graph-court4')
                            ], className="six columns",style={'height':'395px'}),
                        ])
                    ])
                ])
            ])
        ]),
    html.Br(),html.Br(),html.Br(),
    html.Div(children=[
        html.H1(children='''TV Ratings vs Age Group'''),
        html.Br(),
        html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(figure=gen_fig)
        ], className="six columns",style={'height':'395px'}),
        ])
            ])
        ]),
    html.Br(),html.Br(),html.Br(),
    html.Div(children=[
        html.H1(children='''Movies and TV Shows Comparision on Age Group'''),
        html.Br(),
        html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(figure=mt_fig)
        ], className="six columns",style={'height':'395px'}),
        ])
            ])
        ]),
    html.Br(),html.Br(),html.Br(),
    html.Div(children=[
        html.H1(children='''Movie and TV Show Released and Added on Netflix Line Graph'''),
        html.Br(),
        html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(figure=yfig)
        ], className="six columns",style={'height':'395px'}),
        ])
            ])
        ]),
    html.Br(),html.Br(),html.Br(),
    html.Div(children=[
        html.H1(children='''Genres Distribution'''),
        html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(figure=mfig)
        ], className="six columns",style={'height':'395px'}),
                html.Br(),
        ])
            ])
        ]),
    html.Br(),html.Br(),html.Br(),
    dbc.Row([
    dbc.Col(md=6, children=[
            html.Div(children=[
                html.H1(children='''Recommendation System'''),
                html.Br(),
                dcc.Dropdown(
                    id = "inputrec",
                    options=optsrec,
                    value=optsrec[0]
                    ),
                html.Div([
                    html.Div([
                        html.Div([
                            dcc.Graph(id='graph-courtrec')
                            ], className="eight columns",style={'height':'395px'}),
                        html.Br(),
                        html.Br(),
                        ])
                    ])
                ])
            ]),
    dbc.Col(md=6, children=[
            html.Br(),
            cardlast,
            html.Br(),
            ])
            ])
        ])
def topmoviesbyContinent(data):
    maindata = continentstop10(data,'weighted_average_vote')
    maindata["Scores"] = maindata["Scores"].astype(float)
    figtop =  px.histogram(maindata, x=maindata['Titles'], y=maindata['Scores'],color = 'Scores')
    return figtop

@app.callback(Output('graph-court', 'figure'), 
              [Input('input', 'value')])

def update_figure(selected_value):
    data = 'Africa'

    if selected_value == 'AS':
        data = 'Asia'
    elif selected_value == 'OC':
        data = 'Australia'
    elif selected_value == 'EU':
        data = 'Europe'
    elif selected_value == 'NA':
        data = 'North America'
    elif selected_value == 'SA':
        data = 'South America'
    print(data)
    fig = topmoviesbyContinent(data)

    return fig

def topmoviesbyage(data):
    maindata2 = top10byage(data)
    maindata2["Scores"] = maindata2["Scores"].astype(float)
    maindata2["Scores"] = maindata2["Scores"].round(decimals=2)
    figtop = px.histogram(maindata2, x=maindata2['Titles'], y=maindata2['Scores'],color = 'Scores',
                   hover_data=maindata2.columns)
    #figtop = px.histogram(maindata2, x=)
    return figtop

@app.callback(Output('graph-court2', 'figure'), 
              [Input('inputage', 'value')])

def update_figure_age(selected_value):
    data = 'weighted_average_vote'

    if selected_value == 'Average':
        data = 'weighted_average_vote'
    elif selected_value == '0age':
        data = 'weighted_allgenders_0age'
    elif selected_value == '18age':
        data = 'weighted_allgenders_18age'
    elif selected_value == '30age':
        data = 'weighted_allgenders_30age'
    elif selected_value == '45age':
        data = 'weighted_allgenders_45age'
    print(data)
    fig = topmoviesbyage(data)

    return fig

def topmoviesbygenre(data):
    maindata3 = genrestop10(data,'weighted_average_vote')
    maindata3["Scores"] = maindata3["Scores"].astype(float)
    maindata3["Scores"] = maindata3["Scores"].round(decimals=2)
    figtop = px.histogram(maindata3, x=maindata3['Titles'], y=maindata3['Scores'],color='Scores',hover_data=maindata3.columns)
    return figtop

def recomend(data):
    mainrec = content_recommender(data)
    mainrec = mainrec.set_index([pd.Index([10,9,8,7,6,5,4,3,2,1])])
    figrec = px.histogram(mainrec, x=mainrec['title'], y=mainrec.index, color = mainrec.index,
                   hover_data=mainrec.columns)
    return figrec

@app.callback(Output('graph-court3', 'figure'), 
              [Input('inputgenre', 'value')])

def update_figure_genre(selected_value):
    data = 'Dramas'

    if selected_value == 'DR':
        data = 'Dramas'
    elif selected_value == 'IN':
        data = 'International Movies'
    elif selected_value == 'AD':
        data = 'Action & Adventure'
    elif selected_value == 'CO':
        data = 'Comedies'
    elif selected_value == 'CL':
        data = 'Classic Movies'
    elif selected_value == 'TH':
        data = 'Thrillers'
    elif selected_value == 'IM':
        data = 'Independent Movies'
    elif selected_value == 'SF':
        data = 'Sci-Fi & Fantasy'
    elif selected_value == 'CM':
        data = 'Cult Movies'
    elif selected_value == 'SM':
        data = 'Sports Movies'
    elif selected_value == 'RM':
        data = 'Romantic Movies'
    elif selected_value == 'CFM':
        data = 'Children & Family Movies'
    elif selected_value == 'LGB':
        data = 'LGBTQ Movies' 
    elif selected_value == 'HM':
        data = 'Horror Movies'
    elif selected_value == 'SUC':
        data = 'Stand-Up Comedy'
    print(data)
    fig = topmoviesbygenre(data)

    return fig

def topmoviesbygender(data):
    maindata4 = top10gender(data)
    maindata4["Scores"] = maindata4["Scores"].astype(float)
    figtop = px.histogram(maindata4, x=maindata4['Titles'], y=maindata4['Scores'],color = 'Scores')
    return figtop

@app.callback(Output('graph-court4', 'figure'), 
              [Input('input4', 'value')])

def update_figure_genre(selected_value):
    data = 'weighted_males_allages'

    if selected_value == 'M':
        data = 'weighted_males_allages'
    elif selected_value == 'F':
        data = 'weighted_females_allages'
   
    print(data)
    fig = topmoviesbygender(data)

    return fig

@app.callback(
    Output('graph-courtrec', 'figure'),
    [Input('inputrec', 'value')])

def update_reco(selected_value):
    data = 'Transformers: Robots in Disguise'
    if selected_value == 'Transformers':
        data = 'Transformers: Robots in Disguise'
    elif selected_value == 'Inception':
        data = 'Inception'
    elif selected_value == '3 Idiots':
        data = '3 Idiots'
    elif selected_value == 'Gol Maal':
        data = 'Gol Maal'	
    elif selected_value == '3 Days to Kill':
        data = '3 Days to Kill'	
    elif selected_value == 'Race 2':
        data = 'Race 2'	
    elif selected_value == 'Rango':
        data = 'Rango'	
    elif selected_value == 'Rock On!!':
        data = 'Rock On!!'	
        
   
    print(data)
    fig = recomend(data)

    return fig


if __name__ == "__main__":
    app.run_server(debug=False)
