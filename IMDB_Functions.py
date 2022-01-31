import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imdb
import pycountry_convert as pc
import warnings
from scipy import stats 
from scipy.stats import zscore,shapiro
import seaborn as sn
import statsmodels.api as sm
warnings.filterwarnings("ignore")

#%% IMDB_match_star_and_primaryName___remove_Others

def IMDB_match_star_and_primaryName(df):
    # Removing 'tt' from 'titleId' to match it with IDs format provided by python API.
    df['titleId'] = df['titleId'].map(lambda x: x.lstrip('tt'))
     # year_title_star
     # year_title_primaryName
     # concatinate year_title 
     # result = year_title_star == year_title_primaryName (True or False)
    
    df['year_title_star']=df["year"].astype(str) + "_" +df["title"].astype(str) + "_" + df["star"]
    df['year_title_primaryName']=df["year"].astype(str) + "_" +df["title"].astype(str) + "_" + df["primaryName"]
    df['year_title']=df["year"].astype(str) + "_" +df["title"].astype(str)
    # we have star name in separate column against each move, we also have star name in primaryName (along other cast)
    # so by filtering star in primary name will give us one unique row of infomration against that movie.
    # it will return maximum unique values with true IDs.
    
    df['result'] = np.where(df['year_title_star'] == df['year_title_primaryName'], 'True', 'False')
    df=df[df['result'].isin(['True'])]
    
    #print('Shape of dataframe is reduced to = ',df.shape)
    print('Now, do we have any duplicate values in  Year_title ? =  ',df.duplicated(subset=['year_title']).any())
    print('Total number of duplicates in year_title =  ', df.year_title.duplicated(keep=False).sum())
    return df

 #%% IMDB_match_titleID_with_IMDB_API

def IMDB_match_titleID_with_IMDB_API(df):
    # lets treat duplications left by above function now,
    df['Y_T_Duplicate']=df.year_title.duplicated(keep=False)
    fdf = df[df['Y_T_Duplicate']==True] # new dataframe with only duplicate entries to process separately.
    df = df.loc[df['Y_T_Duplicate']==False ] # Drop true values from df as we will treat them in separate DF.
    
    # make long IMBD title to get accurate ID from IMDB API.
    fdf['long_title']=fdf['title'].astype(str) + " (" + fdf["year"].astype(str) + ")"
    ia = imdb.IMDb()
    fdf['Y_T__T_ID'] = fdf['long_title'].apply(lambda x: ia.search_movie(x)[0].movieID)
    
    # now compare new ID with ID in DB
    fdf['ID_vs_ID']=np.where(fdf['titleId'] == fdf['Y_T__T_ID'], True, False)
    
    # Keep only True values
    fdf=fdf.loc[fdf['ID_vs_ID']==True ]
    
    #Reshape the filtered datframe to copy it to our main Dataframe
    fdf=fdf[['titleId', 'title', 'rating', 'region', 'genre', 'released', 'year',
       'month', 'day', 'score', 'director', 'writer', 'star', 'country',
       'budget', 'gross', 'company', 'runtime', 'category', 'nconst',
       'primaryName', 'knownForTitles', 'year_title_star',
       'year_title_primaryName', 'year_title', 'result', 'Y_T_Duplicate']]
    # Combine two dataframes
    df=pd.concat([df,fdf])
    
    # Chekc again if we still have duplicate values in year_title 
    print('Shape of dataframe is further reduced to = ',df.shape)
    
    #print('Now, do we have any duplicate values in  Year_title ? =  ',df.duplicated(subset=['year_title']).any())
    print('Total number of duplicates in  year_title  =  ', df.year_title.duplicated(keep=False).sum())
    
    # Drop unnecessary columns, as we dont have any Duplicate value now.
    df=df.drop(columns=['result','Y_T_Duplicate'])
    
    # save this Dataframe to a CSV file, to load it directly and save (API) Run time next time.
    df.to_csv('NB_1_result_IMDB_UniqueID_10Nov21.csv',index=False)
    return df

#%%  extract_3_geners_from_IMDB

def extract_3_geners_from_IMDB(df,IMDB_title_basics="title.basics.tsv"):
    # load tsv file to dataFrame (Database provided by IMDB)
    title_basics=pd.read_csv(IMDB_title_basics, sep='\t')
    
    # remove tt from tconst and convert it to int, because ID is int type in our previous DF
    title_basics['tconst'] = title_basics['tconst'].map(lambda x: x.lstrip('tt'))
    title_basics['tconst']=title_basics['tconst'].apply(int) 
    
    #####  Vlookup genrese from (above) name_basic to our main DataFrame

    # **  Setp 1: Set ncost as index in boht dataframes (df and name_basics)
    title_basics.set_index('tconst',inplace=True)
    df.set_index('titleId',inplace=True)   #name_basics.reset_index(inplace=True)  ## Toremove index again
    
    # ** Step 2:  make BLANK column name for geners to copy data in main df
    df['3_genres']=""
    
    # ** Setp 3: 
    df['3_genres']=df.index.map(title_basics['genres'])
    
    # ** Step 4: remove tconst and title is from index in boht dataFrames
    title_basics.reset_index(inplace=True)
    df.reset_index(inplace=True)
    return df

#%% country_to_continent

def country_to_continent(country_name):
    country_alpha2 = pc.country_name_to_country_alpha2(country_name)
    country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
    country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
    return country_continent_name
# Function Refrence: https://www.buzzphp.com/posts/get-continent-name-from-country-using-pycountry

#%% V_lookup

#### Function for Vlookup
def V_lookup(df1, df2, df1_index, df2_index, V_from, V_to):
    
    #Step 1 : set index
    df1.set_index(df1_index,inplace=True)
    df2.set_index(df2_index,inplace=True)
    
    #setp2: make blank column 
    df2[V_to]=""
    
    #Step3:
    df2[V_to]=df2.index.map(df1[V_from])
    
    #Setp 4: Remove indexes again 
    df1.reset_index(inplace=True)
    df2.reset_index(inplace=True)
    
# df1 = first DataFrame (Vlookup from) (target DataFrame)
# Df2 = second DataFrame (Vlookup to) (destination DataFrame)

# df1_index = 'string' of column name from df1 (Target Dataframe)
# df2_index = 'string' of column name from df2 (Destination DataFrame)

# V_to = destination column name in df2
# V_from = target column name (from df2)

#%%  countru_to_continent

def countru_to_continent(df):
    # only one NaN in country column, and that too was wierd value so we can drop it for sure
    df=df.dropna(subset=["country"]) 
    
    # Ignoring movies form 'west germany' and 'Yugoslavia'. (Total = 12 movies [including one NaN country])
    df.drop(index=df[df['country'] == 'West Germany'].index, inplace=True)
    df.drop(index=df[df['country'] == 'Federal Republic of Yugoslavia'].index, inplace=True)
    df.drop(index=df[df['country'] == 'Yugoslavia'].index, inplace=True)
    
    # **  convert countries to continent 
    df['continent']=(df["country"].apply(lambda x: country_to_continent(x))) 
    return df

#%% extract_age_of_actor

def extract_age_of_actor(df, IMDB_data="name.basics.tsv"):
    # load tsv file to dataFrame ( Another Database provided by IMDB)
    name_basics=pd.read_csv(IMDB_data, sep='\t')
    
    #####  Vlookup Star age year from name_basic
    # **  Setp 1: Set ncost as index in boht dataframes (df and name_basics)
    name_basics.set_index('nconst',inplace=True)
    df.set_index('nconst',inplace=True)   #name_basics.reset_index(inplace=True)  ## Toremove index again
    # ** Step 2:  make BLANK column name for actor age
    df['star_birthYear']=""
    # ** Setp 3: 
    df['star_birthYear']=df.index.map(name_basics['birthYear'])
    # ** Setp 4: remove nconst as index from boht dataframes
    name_basics.reset_index(inplace=True)
    df.reset_index(inplace=True)
    
    # some actors birth year is not available in any IMDB database and python gets false value "\N" for them
    # convert "\N" to numpy.NaN for further process
    df=df.replace('\\N', np.NaN)
    
    # temporary creat date time values column to get age
    df['starYear']=pd.to_datetime(df['star_birthYear'])
    df['movieYear']=pd.to_datetime(df['year'].apply(str))
    
    #get age
    df['starAge_atMovie']=((df['movieYear']-df['starYear'])/ pd.Timedelta('365 days')).round().astype("Int64")
    
    # as we have age at the time of movie release, so ...
    # delete temporary created columns, we dont need them any way
    df=df.drop(['starYear', 'movieYear','star_birthYear'], axis=1)
    return df

#%% extract_No_of_voters
def extract_No_of_voters(df,IMDB_data3="title.ratings.tsv"):
    # load tsv file to dataFrame
    title_ratings=pd.read_csv(IMDB_data3, sep='\t')
    
    # remove tt from tconst and convert it to int
    title_ratings['tconst'] = title_ratings['tconst'].map(lambda x: x.lstrip('tt'))
    title_ratings['tconst']=title_ratings['tconst'].apply(int)
    
    #####  Vlookup Ratings and Number of votes from title_ratings
    # **  Setp 1: Set ncost as index in boht dataframes (df and name_basics)
    title_ratings.set_index('tconst',inplace=True)
    df.set_index('titleId',inplace=True)   #name_basics.reset_index(inplace=True)  ## Toremove index again
    # ** Step 2:  make BLANK column name for geners to copy data in main df
    df['numVotes']=""
    # ** Setp 3: 
    df['numVotes']=df.index.map(title_ratings['numVotes'])
    # ** Step 4: remove tconst and title is from index in boht dataFrames
    title_ratings.reset_index(inplace=True)
    df.reset_index(inplace=True)
    
    return df

#%% Combine functions
## Function to concatinate column values with "_" in between.
def combine_2(a,b):
    return a.astype(str)+"_"+b.astype(str)
def combine_3(a,b,c):
    return a.astype(str)+"_"+b.astype(str)+"_"+c.astype(str)
def combine_two(a,b):
    return a.astype(str) + "_" + b
## Function to remove TT and convert it to int64
def remove_tt(a):
    b=a.map(lambda x: x.lstrip('tt'))
    return b.apply(int)

#%% sum_of_stars

def sum_of_stars(df,df_IMDB):
        #Filter true title ID from df_IMDB
    titleId_list=list(df['titleId'])

    #Filter main (given) database and get information agianst ONLY true IDS.
    df_IMDB['titleId'] = df_IMDB['titleId'].map(lambda x: x.lstrip('tt'))
    df_IMDB['titleId']=df_IMDB['titleId'].apply(int)

    titleId_list=list(df['titleId']) # list of true IDs which we filtered amove
    df_IMDB=df_IMDB[df_IMDB['titleId'].isin(titleId_list)]

    # Get counts for number of actor, actresses, diterctors, writers, producers. 
    df_counts=df_IMDB.groupby(['titleId','category']).size().reset_index().rename(columns={0:'count'})
    # Refrence: 
    #https://stackoverflow.com/questions/35268817/unique-combinations-of-values-in-selected-columns-in-pandas-data-frame-and-count

    df['year_title_star']=df["year"].astype(str) + "_" +df["title"].astype(str) + "_" + df["star"]
    df_counts['tconst_category']=combine_2(df_counts.titleId , df_counts.category)
    
    # Make temporary variables for Vlookup
    df['tconst_actor'] = combine_two(df.titleId, 'actor')
    df['tconst_actress'] = combine_two(df.titleId, 'actress')
    df['tconst_director'] = combine_two(df.titleId, 'director')
    
    # Vlookup time using above function
    V_lookup(df_counts, df, 'tconst_category', 'tconst_actor', 'count', 'Total_actors')
    
    V_lookup(df_counts, df, 'tconst_category', 'tconst_actress', 'count', 'Total_actress')
    
    V_lookup(df_counts, df, 'tconst_category', 'tconst_director', 'count', 'Total_director')
    
    # Convert np.NaN to 0 becuase according to our DB, there are zero main actors for that move
    # (star or main player might be the actress) .
    df[['Total_actors','Total_actress','Total_director']]=df[['Total_actors','Total_actress','Total_director']].replace(np.NaN, 0)
    df[['Total_actors','Total_actress','Total_director']]=df[['Total_actors','Total_actress','Total_director']].astype(int)
    
    df['actors_sum']=df['Total_actors']+df['Total_actress'] # sum of all actors in movie
    
    #Delete temporary created columns as we dont need them any more ####
    df.drop(['tconst_actor','tconst_actress','tconst_director'], axis=1, inplace=True)
    
    return df

#%% group_year_column

def group_year_column(df):
    year=df['year']
    condition = [year<= 1985, year<=1990, year<=1995, year<=2000, year<=2005, year<=2010, year<=2015, year <=2020 ]
    choice = ["1980-1985", "1986-1990", "1991-1995", "1996-2020", "2001-2005", "2006-2010", "2011-2015", "2016-2020"]
    df['year_group']=np.select(condition, choice)
    
    #Creat CSV for just in case. ;)
    df.to_csv('NB_2_result_IMDB_extractedData_10Nov21.csv',index=False)
    
    return df

#%% max_quantile_replace
def max_quantile_replace(DF,attribute,quantile, replace=False):
    counts=0
    Max_thold=DF[attribute].quantile(quantile)
    #print(quantile*100,' Percentile value for ',attribute,' = ',Max_thold)
    counts+=DF[DF[attribute]>Max_thold].shape[0]
    #print('counts = ',counts)
    
    if replace==True:
        DF[attribute][DF[attribute]>Max_thold]=Max_thold
        #print('Values replaced')
        return DF
################################################        
# Dataframe = DataFrame Name
# Attribute_name = String of attribute name
# Quantile = Quantile values between 0 and 1 (Float type value)
# replace = True if you want to replace values, False by default 
################################################

#%% min_quantile_remove

def min_quantile_remove(DF, attribute,quantile, remove=False):
    counts=0
    Min_thold=DF[attribute].quantile(quantile)
    #print(quantile*100,' Percentile value for ',attribute,' = ',Min_thold)
    counts+=DF[DF[attribute]<Min_thold].shape[0]
    #print('counts = ',counts)
    
    if remove==True:
        DF=DF[DF[attribute]>Min_thold]
        #print('Values Removed')
        return DF
################################################        
# Dataframe = DataFrame Name
# Attribute_name = String of attribute name
# Quantile = Quantile values between 0 and 1 (Float type value)
# remove = True if you want to remove values, False by default wont remove anything
################################################ 
#%% remove_other_outliers(df)
def remove_other_outliers(df):
    list1=['Total_actors','Total_actress', 'Total_director']
    for entry in list1:
        df=max_quantile_replace(df,entry,0.99, replace=True)
    return df

#%% IMDB_boxplot

def IMDB_boxplot(df):
    df=df.dropna()
    cor= df[['year', 'budget', 'runtime', 'starAge_atMovie', 'numVotes',
       'Total_actors', 'Total_actress', 'Total_director', 'gross', 'score']]
    corr = cor.corr().round(2)
    plt.style.use('ggplot')
    attributeNames=cor.columns.tolist()
    M=len(attributeNames)
    red_box = dict(markerfacecolor='r', marker='s',markersize=1)
    
    
    plt.figure(figsize=(5,5))
    plt.boxplot(zscore(cor), attributeNames,flierprops=red_box,patch_artist=True)
    plt.xticks(range(1,M+1), attributeNames,fontsize=10, rotation=90) 
    plt.yticks(fontsize=8)
    plt.title('IMDB:   Standarized Boxplot',fontsize=10)

#%% IMDB_dist_PDF
def IMDB_dist_PDF(df):
    attributeNames=[ 'budget', 'runtime', 'numVotes',
       'Total_actors', 'Total_actress', 'Total_director', 'gross', 'score']
    M=len(attributeNames)
    plt.figure(figsize=(15,5))
    u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
    sn.set()
    for i in range(M):
        attribute_Name = str(attributeNames[i])
        X = df[attribute_Name]
        plt.subplot(int(u),int(v),i+1)
        ab=sn.distplot(X,kde_kws={"color": "r", "lw": 2})
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6) 
        ab.set_xlabel(attribute_Name,fontsize=10)

#%% 
def corl_chart(df,x_size=9,y_size=6,font_size=9):
    
    plt.style.use('fivethirtyeight')
    #plt.rcParams['figure.figsize'] = (10,7)
    plt.rcParams['figure.figsize'] = (x_size,y_size)
    
    # Corelation
    corr = df.corr().round(2)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    
    # code to show only single corner heatmap
    mask = np.zeros_like(corr,dtype=bool)
    mask[np.triu_indices_from(mask)]=True
    corr[mask]=np.nan
    (corr.style.background_gradient(cmap='coolwarm',axis=None,vmin=-1,vmax=1)
     .highlight_null(null_color='#f1f1f1') 
     .set_precision(1))
    
    #display corelation heatmap
    sn.heatmap(corr,cmap="coolwarm",annot=True,linewidths=0.5,linecolor='#f1f1f1')

#%% get_score
def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

#%% make_score_groups
def make_score_groups(df):
    temp=df['score']
    condition = [temp<= 1, temp<=2, temp<=3, temp<=4, temp<=5, temp<=6, temp<=7, temp <=8, temp <=9, temp <=10 ]
    choice = ["0-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8","8-9","9-10"]
    df['score_label']=np.select(condition, choice)
    return df

#%%



    
    





































































































































