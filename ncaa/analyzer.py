import pandas as pd
import numpy as np
import pylab as P
import matplotlib
from sklearn.ensemble import RandomForestClassifier 
import csv

#Change the max rows
pd.set_option('display.width', 1000)

#Switches
year = 2015 #<-- which year are we trying to predict

#Load in regular season data
df = pd.read_csv('data/RegularSeasonDetailedResults.csv', header = 0)

#Load in tourney data
tourney = pd.read_csv('data/TourneyDetailedResults.csv', header = 0) 

#Load in Team Information
teams = pd.read_csv('data/Teams.csv', header = 0)

#Load in seed information
seeds = pd.read_csv('data/TourneySeeds.csv', header = 0)

#------------------------    EXPLANATION OF VARIABLES    -------------------------------------
#Timing information
  #Season - year game was played
  #Daynum - day number of tournament
  #Numot - number of overtime periods

#Game Results
  #Wteam - winning team ID number
  #Lteam - losing team ID number
  #Wscore - winning team score
  #Lscore -losing team score

#Location information
  #Wloc - H if winning team was at home, A if winning time was visiting, N for neutral

#Detailed scoring information -- W for winner, L for loser
  #Wfgm, Wfga    - number of field goals (non free-throws) made, attempted
  #Wfgm3, Wfga3  - number of 3-pointers made, attempted
  #Wftm, Wfta - number of free throws made, attempted
  #Wor, Wdr - number of offensive (defensive) rebounds
  #Wast - number of assists
  #Wto - number of turnovers
  #Wstl - number of steals
  #Wblk -number of blocks
  #Wpf -number of personal fouls

#------------------------   FIGURE OUT BEST TEAMS -------------------------------------
#If a team has been in the top-32 for at least 2 of the last 3 years running, they get class=1
for k in [df, tourney]:
  for j in range(1101, 1465): 
    number = 0
    for i in range(year-3, year):
      nWins = tourney[ (tourney.Season == i) & ((tourney.Wteam == j) | (tourney.Lteam == j))][['Season']].count()
      if (int(nWins) > 1): number += 1
    if (number >= 2): 
      teams.loc[ (teams.Team_Id == j) & (teams.Team_Class == 2), 'Team_Class'] = 1
    k.loc[ (k.Wteam == j), 'WteamClass'] = teams[ (teams.Team_Id == j)]['Team_Class'].mean()
    k.loc[ (k.Lteam == j), 'LteamClass'] = teams[ (teams.Team_Id == j)]['Team_Class'].mean()

#------------------------   DEFINE TEAM 1 AND TEAM 2 FOR EACH GAME -------------------------------------
for i in range(1101, 1465):
  df.loc[ (df.Wteam == i) & (df.Wteam > df.Lteam), 'Team2' ] = df.Wteam
  df.loc[ (df.Wteam == i) & (df.Wteam > df.Lteam), 'Team1' ] = df.Lteam
  df.loc[ (df.Wteam == i) & (df.Wteam < df.Lteam), 'Team2' ] = df.Lteam
  df.loc[ (df.Wteam == i) & (df.Wteam < df.Lteam), 'Team1' ] = df.Wteam
  df.loc[ (df.Wteam == i) & (df.Wteam > df.Lteam), 'Team2score' ] = df.Wscore
  df.loc[ (df.Wteam == i) & (df.Wteam > df.Lteam), 'Team1score' ] = df.Lscore
  df.loc[ (df.Wteam == i) & (df.Wteam < df.Lteam), 'Team2score' ] = df.Lscore
  df.loc[ (df.Wteam == i) & (df.Wteam < df.Lteam), 'Team1score' ] = df.Wscore
  df.loc[ (df.Wteam == i) & (df.Wteam > df.Lteam), 'Team2class' ] = df.WteamClass
  df.loc[ (df.Wteam == i) & (df.Wteam > df.Lteam), 'Team1class' ] = df.LteamClass
  df.loc[ (df.Wteam == i) & (df.Wteam < df.Lteam), 'Team2class' ] = df.LteamClass
  df.loc[ (df.Wteam == i) & (df.Wteam < df.Lteam), 'Team1class' ] = df.WteamClass
  df.loc[ (df.Wteam == i) & (df.Wteam < df.Lteam) & (df.Wloc == 'H'), 'Team1isHome' ] = 1
  df.loc[ (df.Wteam == i) & (df.Wteam > df.Lteam) & (df.Wloc == 'H'), 'Team1isHome' ] = 0
  df.loc[ (df.Wteam == i) & (df.Wteam < df.Lteam) & (df.Wloc == 'A'), 'Team1isHome' ] = 0
  df.loc[ (df.Wteam == i) & (df.Wteam > df.Lteam) & (df.Wloc == 'A'), 'Team1isHome' ] = 1
  df.loc[ (df.Wteam == i) & (df.Wloc == 'N'), 'Team1isHome' ] = 2
df['ClassDiff'] = df.Team1class - df.Team2class
df = df.drop( [ 'Wteam', 'Lteam', 'Wscore', 'Lscore', 'WteamClass', 'LteamClass', 'Team1class', 'Team2class' ], axis=1) 

#------------------------    FOM FOR EACH REGULAR GAME    -------------------------------------
df.loc[ (df.Team1score > df.Team2score), 'FOM1' ] = (df.Team1score / (df.Team1score + df.Team2score)) 
df.loc[ (df.Team1score > df.Team2score) & (df.Team1score - df.Team2score > 4), 'FOM1' ] += .02*(df.Team1score - df.Team2score - 4)
df.loc[ (df.Team1score > df.Team2score), 'FOM2' ] = 1-df.FOM1
df.loc[ (df.Team1score < df.Team2score), 'FOM2' ] = (df.Team2score / (df.Team1score + df.Team2score))
df.loc[ (df.Team1score < df.Team2score) & (df.Team2score - df.Team1score > 4), 'FOM2' ] += .02*(df.Team2score - df.Team1score - 4)
df.loc[ (df.Team1score < df.Team2score), 'FOM1' ] = 1-df.FOM2
df.loc[ (df.FOM1 > 0.95), 'FOM1' ] = 0.95
df.loc[ (df.FOM2 > 0.95), 'FOM2' ] = 0.95
df.loc[ (df.FOM1 < 0.05), 'FOM1' ] = 0.05
df.loc[ (df.FOM2 < 0.05), 'FOM2' ] = 0.05

#------------------------    CALC AFOM FOR EACH TEAM  -------------------------------------
for i in range(1101,1465):
  teams.loc[ (teams.Team_Id == i), 'AFOM1'] =  (df[ (df.Season == year) & (df.Team1 == i)]['FOM1'].sum() + df[ (df.Season == year) & (df.Team2 == i)]['FOM2'].sum())/df[ (df.Season == year) & ((df.Team1 == i) | (df.Team2 == i))]['FOM1'].count()
  df.loc[ (df.Team1 == i), 'Team1AFOM1']  = teams[ (teams.Team_Id == i)]['AFOM1'].mean()
  df.loc[ (df.Team2 == i), 'Team2AFOM1']  = teams[ (teams.Team_Id == i)]['AFOM1'].mean()

##---------------------  NOW CALCULATE PRED FOM FOR EACH REGULAR SEASON GAME ------------------------
#Make base prediction
df.loc[(abs(df.ClassDiff) < 4), 'PFOM1'] = df.Team1AFOM1/(df.Team1AFOM1+df.Team2AFOM1)
#Now be more confident in prediction
df.loc[ (df.PFOM1 > 0.5), 'Error'] = (df.FOM1 - df.PFOM1)/df.PFOM1
df.loc[ (df.PFOM1 < 0.5), 'Error'] = -(df.FOM1 - df.PFOM1)/df.PFOM1
corr = df[ (df.ClassDiff == 0) & ((df.Season == year) | (df.Season == year - 1))& (df.Team1AFOM1 > .2) & (df.Team2AFOM1 < .8) ][['Error']].mean()
df.loc[ (df.PFOM1 > 0.54), 'PFOM1'] *= 1.+float(corr)
df.loc[ (df.PFOM1 < 0.46), 'PFOM1'] = 1.-(1.-df.PFOM1)*(1.+float(corr))
#Finally, account for differences in class
df.loc[ (df.ClassDiff < 0), 'PFOM1'] = df.PFOM1**(1/(2*abs(df.ClassDiff)))
df.loc[ (df.ClassDiff > 0), 'PFOM1'] = 1.-((1.-df.PFOM1)**(1/(2*abs(df.ClassDiff))))

#--------------------- USE THIS TO MAKE PREDICTIONS FOR 2015 TOURNEY
for i in range(1101, 1465):
  tourney.loc[ (tourney.Wteam == i) & (tourney.Wteam > tourney.Lteam), 'Team2' ] = tourney.Wteam
  tourney.loc[ (tourney.Wteam == i) & (tourney.Wteam > tourney.Lteam), 'Team1' ] = tourney.Lteam
  tourney.loc[ (tourney.Wteam == i) & (tourney.Wteam < tourney.Lteam), 'Team2' ] = tourney.Lteam
  tourney.loc[ (tourney.Wteam == i) & (tourney.Wteam < tourney.Lteam), 'Team1' ] = tourney.Wteam
  tourney.loc[ (tourney.Wteam == i) & (tourney.Wteam > tourney.Lteam), 'Team2score' ] = tourney.Wscore
  tourney.loc[ (tourney.Wteam == i) & (tourney.Wteam > tourney.Lteam), 'Team1score' ] = tourney.Lscore
  tourney.loc[ (tourney.Wteam == i) & (tourney.Wteam < tourney.Lteam), 'Team2score' ] = tourney.Lscore
  tourney.loc[ (tourney.Wteam == i) & (tourney.Wteam < tourney.Lteam), 'Team1score' ] = tourney.Wscore
  tourney.loc[ (tourney.Wteam == i) & (tourney.Wteam > tourney.Lteam), 'Team2class' ] = tourney.WteamClass
  tourney.loc[ (tourney.Wteam == i) & (tourney.Wteam > tourney.Lteam), 'Team1class' ] = tourney.LteamClass
  tourney.loc[ (tourney.Wteam == i) & (tourney.Wteam < tourney.Lteam), 'Team2class' ] = tourney.LteamClass
  tourney.loc[ (tourney.Wteam == i) & (tourney.Wteam < tourney.Lteam), 'Team1class' ] = tourney.WteamClass
  tourney.loc[ (tourney.Wteam == i) & (tourney.Wteam < tourney.Lteam) & (tourney.Wloc == 'H'), 'Team1isHome' ] = 1
  tourney.loc[ (tourney.Wteam == i) & (tourney.Wteam > tourney.Lteam) & (tourney.Wloc == 'H'), 'Team1isHome' ] = 0
  tourney.loc[ (tourney.Wteam == i) & (tourney.Wteam < tourney.Lteam) & (tourney.Wloc == 'A'), 'Team1isHome' ] = 0
  tourney.loc[ (tourney.Wteam == i) & (tourney.Wteam > tourney.Lteam) & (tourney.Wloc == 'A'), 'Team1isHome' ] = 1
  tourney.loc[ (tourney.Wteam == i) & (tourney.Wloc == 'N'), 'Team1isHome' ] = 2
tourney['ClassDiff'] = tourney.Team1class - tourney.Team2class
tourney = tourney.drop( [ 'Wteam', 'Lteam', 'Wscore', 'Lscore', 'WteamClass', 'LteamClass', 'Team1class', 'Team2class' ], axis=1) 

#------------------------    FOM FOR EACH REGULAR GAME    -------------------------------------
tourney.loc[ (tourney.Team1score > tourney.Team2score), 'FOM1' ] = (tourney.Team1score / (tourney.Team1score + tourney.Team2score)) 
tourney.loc[ (tourney.Team1score > tourney.Team2score) & (tourney.Team1score - tourney.Team2score > 4), 'FOM1' ] += .02*(tourney.Team1score - tourney.Team2score - 4)
tourney.loc[ (tourney.Team1score > tourney.Team2score), 'FOM2' ] = 1-tourney.FOM1
tourney.loc[ (tourney.Team1score < tourney.Team2score), 'FOM2' ] = (tourney.Team2score / (tourney.Team1score + tourney.Team2score))
tourney.loc[ (tourney.Team1score < tourney.Team2score) & (tourney.Team2score - tourney.Team1score > 4), 'FOM2' ] += .02*(tourney.Team2score - tourney.Team1score - 4)
tourney.loc[ (tourney.Team1score < tourney.Team2score), 'FOM1' ] = 1-tourney.FOM2
tourney.loc[ (tourney.FOM1 > 0.95), 'FOM1' ] = 0.95
tourney.loc[ (tourney.FOM2 > 0.95), 'FOM2' ] = 0.95
tourney.loc[ (tourney.FOM1 < 0.05), 'FOM1' ] = 0.05
tourney.loc[ (tourney.FOM2 < 0.05), 'FOM2' ] = 0.05

#Load in teams AFOM
for i in range(1101,1465):
  tourney.loc[ (tourney.Team1 == i), 'Team1AFOM1']  = teams[ (teams.Team_Id == i)]['AFOM1'].mean()
  tourney.loc[ (tourney.Team2 == i), 'Team2AFOM1']  = teams[ (teams.Team_Id == i)]['AFOM1'].mean()

##---------------------  NOW CALCULATE PRED FOM FOR EACH REGULAR SEASON GAME ------------------------
#Make base prediction
tourney.loc[(abs(tourney.ClassDiff) < 4), 'PFOM1'] = tourney.Team1AFOM1/(tourney.Team1AFOM1+tourney.Team2AFOM1)
#Now be more confident in prediction
tourney.loc[ (tourney.PFOM1 > 0.5), 'Error'] = (tourney.FOM1 - tourney.PFOM1)/tourney.PFOM1
tourney.loc[ (tourney.PFOM1 < 0.5), 'Error'] = -(tourney.FOM1 - tourney.PFOM1)/tourney.PFOM1
corr = tourney[ (tourney.ClassDiff == 0) & (tourney.Season == year) & (tourney.Team1AFOM1 > .2) & (tourney.Team2AFOM1 < .8) ][['Error']].mean()
tourney.loc[ (tourney.PFOM1 > 0.54), 'PFOM1'] *= 1.+float(corr)
tourney.loc[ (tourney.PFOM1 < 0.46), 'PFOM1'] = 1.-(1.-tourney.PFOM1)*(1.+float(corr))
#Finally, account for differences in class when early in the tournament
tourney.loc[ (tourney.Daynum < 140) & (tourney.ClassDiff < 0), 'PFOM1'] = tourney.PFOM1**(1/(2*abs(tourney.ClassDiff)))
tourney.loc[ (tourney.Daynum < 140) & (tourney.ClassDiff > 0), 'PFOM1'] = 1.-((1.-tourney.PFOM1)**(1/(2*abs(tourney.ClassDiff))))

#Now we're going to make this prediction using machine learning
train = df[(df.Season == 2015) | (df.Season == 2014)][['FOM1','Team1AFOM1','Team2AFOM1','ClassDiff']]
train_data = train.values
tourney['ClassDiff'] = tourney.ClassDiff * (tourney.Daynum < 140)

forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data[0::,1::],train_data[0::,0].astype(str))
test = tourney[tourney.Season == 2015][['FOM1','Team1AFOM1','Team2AFOM1','ClassDiff']]
test_data = test.values
output = forest.predict(test_data[0::,1::]).astype(float)
tourney.loc[ (tourney.Season == 2015), 'CPFOM'] = output
tourney.loc[ (tourney.Season == 2015), 'Myerr'] = (tourney.PFOM1 - tourney.FOM1)/tourney.FOM1
tourney.loc[ (tourney.Season == 2015), 'Coerr'] = (tourney.CPFOM - tourney.FOM1)/tourney.FOM1
tourney.loc[ (tourney.Season == 2015) & ((tourney.CPFOM > .3) | (tourney.CPFOM < .7)), 'PFOM'] = tourney.PFOM1
tourney.loc[ (tourney.Season == 2015) & ((tourney.CPFOM < .3) | (tourney.CPFOM > .7)), 'PFOM'] = tourney.CPFOM

#---------------------  NOW CALCULATE BASED ON SEED -- signed (SEED1 - Seed2)           ------------------------
for i in range(1101,1465):
  tourney.loc[ (tourney.Team1 == i), 'Team1Seed']  = seeds[ (seeds.Season == year) & (seeds.Team == i)]['Seed'].mean()
  tourney.loc[ (tourney.Team2 == i), 'Team2Seed']  = seeds[ (seeds.Season == year) & (seeds.Team == i)]['Seed'].mean()
tourney['SeedDiff'] = tourney['Team1Seed'] - tourney['Team2Seed']
print tourney[tourney.Season == year][['Team1','Team2','ClassDiff','Team1score','Team2score','FOM1','PFOM', 'SeedDiff','Team1isHome']].head(30)

#---------------------  NOW CALCULATE HOMEFIELD ADVANTAGE  ----------------------------------------------------


#--------------------  NOW CALCULATE WINNINGNESS PREDICTIONS ----------------------------------------------------

