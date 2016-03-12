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

#------------------------    GENERAL OVERVIEW OF METHOD    -------------------------------------
#  1) We train on the 2012-2015 regular season data in order to collect all our variables
#     (except the seeds, obviously)
#  2) We test #1 on the tourney data from 2001-present
#  3) Then we use the 2001-2011 tourney data to train our high-level variables on the 
#     final prediction
#  4) So that we can test the final prediction on everything. 

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
  #Wast - number of assists
  #Wstl - number of steals
  #Wor, Wdr - number of offensive (defensive) rebounds
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

#------------------------    CALC AFOM AND STDDEV FOR EACH TEAM  -------------------------------------
for i in range(1101,1465):
  a = df[ (df.Season == year) & (df.Team1 == i)]['FOM1'].values
  a = np.append(a, df[ (df.Season == year) & (df.Team2 == i)]['FOM2'].values)
  b = df[ (df.Season == year) & (df.Team1 == i)]['FOM2'].values
  b = np.append(a, df[ (df.Season == year) & (df.Team2 == i)]['FOM1'].values)
  teams.loc[ (teams.Team_Id == i), 'AFOM1'] = np.mean(a)
  teams.loc[ (teams.Team_Id == i), 'OPFOM1'] = np.mean(b)
  teams.loc[ (teams.Team_Id == i), 'AFOMSD'] = np.std(a)
  df.loc[ (df.Team1 == i), 'Team1AFOM1']  = teams[ (teams.Team_Id == i)]['AFOM1'].mean()
  df.loc[ (df.Team2 == i), 'Team2AFOM1']  = teams[ (teams.Team_Id == i)]['AFOM1'].mean()
  df.loc[ (df.Team1 == i), 'Team1OPFOM1']  = teams[ (teams.Team_Id == i)]['OPFOM1'].mean()
  df.loc[ (df.Team2 == i), 'Team2OPFOM1']  = teams[ (teams.Team_Id == i)]['OPFOM1'].mean()
  df.loc[ (df.Team1 == i), 'Team1AFOMSD'] = teams[ (teams.Team_Id == i)]['AFOMSD'].mean()
  df.loc[ (df.Team2 == i), 'Team2AFOMSD'] = teams[ (teams.Team_Id == i)]['AFOMSD'].mean()
df['AFOMSD'] = df.Team1AFOMSD + df.Team2AFOMSD
df['OPFOMDIF'] = df.Team1OPFOM1 - df.Team2OPFOM1

##---------------------  NOW CALCULATE PRED FOM FOR EACH REGULAR SEASON GAME ------------------------
#Make base prediction
df.loc[(abs(df.ClassDiff) < 4), 'PFOM1'] = df.Team1AFOM1/(df.Team1AFOM1+df.Team2AFOM1)
#Now be more confident in prediction
df.loc[ (df.PFOM1 > 0.5), 'Error'] = (df.FOM1 - df.PFOM1)/df.PFOM1
df.loc[ (df.PFOM1 < 0.5), 'Error'] = -(df.FOM1 - df.PFOM1)/df.PFOM1
corr = df[ (df.ClassDiff == 0) & ((df.Season == year) | (df.Season == year - 1)) & (df.Team1AFOM1 > .2) & (df.Team2AFOM1 < .8) ][['Error']].mean()
df.loc[ (df.PFOM1 > 0.54), 'PFOM1'] *= 1.+float(corr)
df.loc[ (df.PFOM1 < 0.46), 'PFOM1'] = 1.-(1.-df.PFOM1)*(1.+float(corr))
#Finally, account for differences in class
df.loc[ (df.ClassDiff < 0), 'PFOM1'] = df.PFOM1**(1/(2*abs(df.ClassDiff)))
df.loc[ (df.ClassDiff > 0), 'PFOM1'] = 1.-((1.-df.PFOM1)**(1/(2*abs(df.ClassDiff))))

#---------------------  NOW CALCULATE FREE THROW SCORE FOR EACH TEAM           ------------------------
for i in range(1101, 1465):
  Ftm  = df.loc[ (df.Season == year) & (df.Wteam == i)]['Wftm'].sum()
  Fta  = df.loc[ (df.Season == year) & (df.Wteam == i)]['Wfta'].sum()
  Ftm += df.loc[ (df.Season == year) & (df.Lteam == i)]['Lftm'].sum()
  Fta += df.loc[ (df.Season == year) & (df.Lteam == i)]['Lfta'].sum()
  nFouls = df.loc[ (df.Season == year) & (df.Wteam == i)]['Wpf'].sum()
  nFouls += df.loc[ (df.Season == year) & (df.Lteam == i)]['Lpf'].sum()
  count = df.loc[ (df.Season == year) & (df.Wteam == i)]['Wpf'].count()
  count += df.loc[ (df.Season == year) & (df.Lteam == i)]['Lpf'].count()
  if (Fta > 0): teams.loc[ (teams.Team_Id == i), 'FTP' ] = float(Ftm)/float(Fta)
  if (count > 0): teams.loc[ (teams.Team_Id == i), 'nFouls' ] = nFouls/count
  df.loc[ (df.Team1 == i), 'Team1FTP'] = teams[ (teams.Team_Id == i)]['FTP'].mean()
  df.loc[ (df.Team2 == i), 'Team2FTP'] = teams[ (teams.Team_Id == i)]['FTP'].mean()
  df.loc[ (df.Team1 == i), 'Team1NF']  = teams[ (teams.Team_Id == i)]['nFouls'].mean()
  df.loc[ (df.Team2 == i), 'Team2NF']  = teams[ (teams.Team_Id == i)]['nFouls'].mean()
df['FTScore'] = df.Team1FTP * df.Team2NF - df.Team2FTP * df.Team1NF

#---------------------  NOW CALCULATE OFFENSIVE & DEFENSIVE SCORE FOR EACH TEAM           ------------------------
for i in range(1101, 1465):
  Fgm    = df.loc[ (df.Season == year) & (df.Wteam == i)]['Wfgm'].sum()
  Fga    = df.loc[ (df.Season == year) & (df.Wteam == i)]['Wfga'].sum()
  Fgm   += df.loc[ (df.Season == year) & (df.Lteam == i)]['Lfgm'].sum()
  Fga   += df.loc[ (df.Season == year) & (df.Lteam == i)]['Lfga'].sum()
  if (Fga > 0): teams.loc[ (teams.Team_Id == i), 'FGP' ] = float(Fgm)/float(Fga)
  teams.loc[ (teams.Team_Id == i), 'FGA' ] = Fga
  Fgm3    = df.loc[ (df.Season == year) & (df.Wteam == i)]['Wfgm3'].sum()
  Fga3    = df.loc[ (df.Season == year) & (df.Wteam == i)]['Wfga3'].sum()
  Fgm3   += df.loc[ (df.Season == year) & (df.Lteam == i)]['Lfgm3'].sum()
  Fga3   += df.loc[ (df.Season == year) & (df.Lteam == i)]['Lfga3'].sum()
  if (Fga3 > 0): teams.loc[ (teams.Team_Id == i), 'FGP3' ] = float(Fgm3)/float(Fga3)
  teams.loc[ (teams.Team_Id == i), 'FGA3' ] = Fga3
  #Turnovers
  df.loc[ (df.Wteam == i) & (df.Wteam > df.Lteam), 'Team2TO'] = df.Wto
  df.loc[ (df.Wteam == i) & (df.Wteam > df.Lteam), 'Team1TO'] = df.Lto
  df.loc[ (df.Wteam == i) & (df.Wteam < df.Lteam), 'Team1TO'] = df.Wto
  df.loc[ (df.Wteam == i) & (df.Wteam < df.Lteam), 'Team2TO'] = df.Lto
  #Steals
  df.loc[ (df.Wteam == i) & (df.Wteam > df.Lteam), 'Team2ST'] = df.Wstl
  df.loc[ (df.Wteam == i) & (df.Wteam > df.Lteam), 'Team1ST'] = df.Lstl
  df.loc[ (df.Wteam == i) & (df.Wteam < df.Lteam), 'Team1ST'] = df.Wstl
  df.loc[ (df.Wteam == i) & (df.Wteam < df.Lteam), 'Team2ST'] = df.Lstl
  #Assists
  df.loc[ (df.Wteam == i) & (df.Wteam > df.Lteam), 'Team2AST'] = df.Wast
  df.loc[ (df.Wteam == i) & (df.Wteam > df.Lteam), 'Team1AST'] = df.Last
  df.loc[ (df.Wteam == i) & (df.Wteam < df.Lteam), 'Team1AST'] = df.Wast
  df.loc[ (df.Wteam == i) & (df.Wteam < df.Lteam), 'Team2AST'] = df.Last
  #Blocks
  df.loc[ (df.Wteam == i) & (df.Wteam > df.Lteam), 'Team2BLK'] = df.Wblk
  df.loc[ (df.Wteam == i) & (df.Wteam > df.Lteam), 'Team1BLK'] = df.Lblk
  df.loc[ (df.Wteam == i) & (df.Wteam < df.Lteam), 'Team1BLK'] = df.Wblk
  df.loc[ (df.Wteam == i) & (df.Wteam < df.Lteam), 'Team2BLK'] = df.Lblk
  #Offensive Rebounds
  df.loc[ (df.Wteam == i) & (df.Wteam > df.Lteam), 'Team2OR'] = df.Wor
  df.loc[ (df.Wteam == i) & (df.Wteam > df.Lteam), 'Team1OR'] = df.Lor
  df.loc[ (df.Wteam == i) & (df.Wteam < df.Lteam), 'Team1OR'] = df.Wor
  df.loc[ (df.Wteam == i) & (df.Wteam < df.Lteam), 'Team2OR'] = df.Lor
  #Offensive Rebounds
  df.loc[ (df.Wteam == i) & (df.Wteam > df.Lteam), 'Team2DR'] = df.Wdr
  df.loc[ (df.Wteam == i) & (df.Wteam > df.Lteam), 'Team1DR'] = df.Ldr
  df.loc[ (df.Wteam == i) & (df.Wteam < df.Lteam), 'Team1DR'] = df.Wdr
  df.loc[ (df.Wteam == i) & (df.Wteam < df.Lteam), 'Team2DR'] = df.Ldr

#Now fill teams
for i in range(1101, 1465):
  a = df[ (df.Season == year) & (df.Team1 == i)]['Team1TO'].values
  a = np.append(a, df[ (df.Season == year) & (df.Team2 == i)]['Team2TO'].values)
  teams.loc[ (teams.Team_Id == i), 'NTO'] = np.mean(a)
  z = df[ (df.Season == year) & (df.Team1 == i)]['Team1ST'].values
  z = np.append(a, df[ (df.Season == year) & (df.Team2 == i)]['Team2ST'].values)
  teams.loc[ (teams.Team_Id == i), 'NST'] = np.mean(z)
  a = df[ (df.Season == year) & (df.Team1 == i)]['Team1AST'].values
  a = np.append(a, df[ (df.Season == year) & (df.Team2 == i)]['Team2AST'].values)
  teams.loc[ (teams.Team_Id == i), 'NAST'] = np.mean(a)
  a = df[ (df.Season == year) & (df.Team1 == i)]['Team1BLK'].values
  a = np.append(a, df[ (df.Season == year) & (df.Team2 == i)]['Team2BLK'].values)
  teams.loc[ (teams.Team_Id == i), 'NBLK'] = np.mean(a)
  a = df[ (df.Season == year) & (df.Team1 == i)]['Team1OR'].values
  a = np.append(a, df[ (df.Season == year) & (df.Team2 == i)]['Team2OR'].values)
  teams.loc[ (teams.Team_Id == i), 'NOR'] = np.mean(a)
  a = df[ (df.Season == year) & (df.Team1 == i)]['Team1DR'].values
  a = np.append(a, df[ (df.Season == year) & (df.Team2 == i)]['Team2DR'].values)
  teams.loc[ (teams.Team_Id == i), 'NDR'] = np.mean(a)
#Now Fill dataFrame
for i in range(1101, 1465):
  df.loc[ (df.Team1 == i), 'TO1']   = teams[ (teams.Team_Id == i)]['NTO'].mean()
  df.loc[ (df.Team2 == i), 'TO2']   = teams[ (teams.Team_Id == i)]['NTO'].mean()
  df.loc[ (df.Team1 == i), 'ST1']   = teams[ (teams.Team_Id == i)]['NST'].mean()
  df.loc[ (df.Team2 == i), 'ST2']   = teams[ (teams.Team_Id == i)]['NST'].mean()
  df.loc[ (df.Team1 == i), 'AST1']  = teams[ (teams.Team_Id == i)]['NAST'].mean()
  df.loc[ (df.Team2 == i), 'AST2']  = teams[ (teams.Team_Id == i)]['NAST'].mean()
  df.loc[ (df.Team1 == i), 'BLK1']  = teams[ (teams.Team_Id == i)]['NBLK'].mean()
  df.loc[ (df.Team2 == i), 'BLK2']  = teams[ (teams.Team_Id == i)]['NBLK'].mean()
  df.loc[ (df.Team1 == i), 'OR1']   = teams[ (teams.Team_Id == i)]['NOR'].mean()
  df.loc[ (df.Team2 == i), 'OR2']   = teams[ (teams.Team_Id == i)]['NOR'].mean()
  df.loc[ (df.Team1 == i), 'DR1']   = teams[ (teams.Team_Id == i)]['NDR'].mean()
  df.loc[ (df.Team2 == i), 'DR2']   = teams[ (teams.Team_Id == i)]['NDR'].mean()
  df.loc[ (df.Team1 == i), 'FGP1']  = teams[ (teams.Team_Id == i)]['FGP'].mean()
  df.loc[ (df.Team2 == i), 'FGP2']  = teams[ (teams.Team_Id == i)]['FGP'].mean()
  df.loc[ (df.Team1 == i), 'FGA1']  = teams[ (teams.Team_Id == i)]['FGA'].mean()
  df.loc[ (df.Team2 == i), 'FGA2']  = teams[ (teams.Team_Id == i)]['FGA'].mean()
  df.loc[ (df.Team1 == i), 'FGP31'] = teams[ (teams.Team_Id == i)]['FGP3'].mean()
  df.loc[ (df.Team2 == i), 'FGP32'] = teams[ (teams.Team_Id == i)]['FGP3'].mean()
  df.loc[ (df.Team1 == i), 'FGA31'] = teams[ (teams.Team_Id == i)]['FGA3'].mean()
  df.loc[ (df.Team2 == i), 'FGA32'] = teams[ (teams.Team_Id == i)]['FGA3'].mean()
df['TO_DIFF'] = df.TO1 - df.TO2
df['ST_DIFF'] = df.ST1 - df.ST2
df['AST_DIFF'] = df.AST1 - df.AST2
df['BLK_DIFF'] = df.BLK1 - df.BLK2
df['OR_DIFF'] = df.OR1 - df.OR2
df['DR_DIFF'] = df.DR1 - df.DR2
df['FGP_DIFF'] = df.FGP1 - df.FGP2
df['FGA_DIFF'] = df.FGA1 - df.FGA2
df['FGP3_DIFF'] = df.FGP31 - df.FGP32
df['FGA3_DIFF'] = df.FGA31 - df.FGA32

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
#tourney = tourney.drop( [ 'Wteam', 'Lteam', 'Wscore', 'Lscore', 'WteamClass', 'LteamClass', 'Team1class', 'Team2class' ], axis=1) 

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
  tourney.loc[ (tourney.Team1 == i), 'Team1OPFOM1']  = teams[ (teams.Team_Id == i)]['OPFOM1'].mean()
  tourney.loc[ (tourney.Team2 == i), 'Team2OPFOM1']  = teams[ (teams.Team_Id == i)]['OPFOM1'].mean()
  tourney.loc[ (tourney.Team1 == i), 'Team1AFOMSD'] = teams[ (teams.Team_Id == i)]['AFOMSD'].mean()
  tourney.loc[ (tourney.Team2 == i), 'Team2AFOMSD'] = teams[ (teams.Team_Id == i)]['AFOMSD'].mean()
tourney['AFOMSD'] = tourney.Team1AFOMSD + tourney.Team2AFOMSD
tourney['OPFOMDIF'] = tourney.Team1OPFOM1 - tourney.Team2OPFOM1
tourney.loc[ tourney.AFOMSD.isnull(), 'AFOMSD' ] = 0.1

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
train = df[(df.Season >= 2012) & (df.Season < 2016)][['FOM1','Team1AFOM1','Team2AFOM1','ClassDiff']]
train_data = train.values
tourney['ClassDiff'] = tourney.ClassDiff * (tourney.Daynum < 140)
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data[0::,1::],train_data[0::,0].astype(str))
test = tourney[tourney.Season >= 2001][['FOM1','Team1AFOM1','Team2AFOM1','ClassDiff']]
test_data = test.values
output = forest.predict(test_data[0::,1::]).astype(float)
tourney.loc[ (tourney.Season >= 2001), 'CPFOM'] = output
tourney.loc[ (tourney.Season >= 2001), 'Myerr'] = (tourney.PFOM1 - tourney.FOM1)/tourney.FOM1
tourney.loc[ (tourney.Season >= 2001), 'Coerr'] = (tourney.CPFOM - tourney.FOM1)/tourney.FOM1
tourney.loc[ (tourney.Season >= 2001) & ((tourney.CPFOM > .3) | (tourney.CPFOM < .7)), 'PFOM'] = tourney.PFOM1
tourney.loc[ (tourney.Season >= 2001) & ((tourney.CPFOM < .3) | (tourney.CPFOM > .7)), 'PFOM'] = tourney.CPFOM
fom_vec = []
for i in range(1101,1465):
  for j in range(i+1, 1465): 
    Team1AFOM1 = teams[ (teams.Team_Id == i)]['AFOM1'].mean()
    Team2AFOM1 = teams[ (teams.Team_Id == j)]['AFOM1'].mean()
    Team1_Class  = teams[ (teams.Team_Id == i)]['Team_Class'].mean()
    Team2_Class  = teams[ (teams.Team_Id == j)]['Team_Class'].mean()
    ClassDiff = Team1_Class-Team2_Class
    if ((Team1AFOM1 != Team1AFOM1) or (Team2AFOM1 != Team2AFOM1)):
      fom_vec.append(0)
      continue
    CPFOM    = forest.predict([Team1AFOM1, Team2AFOM1, ClassDiff])
    myFOM    = Team1AFOM1/(Team1AFOM1+Team2AFOM1)
    if (myFOM > 0.54): myFOM *= 1.+float(corr)
    if (myFOM < 0.46): myFOM *= 1.-float(corr)
    if (ClassDiff < 0): myFOM = myFOM**(1/(2*abs(ClassDiff)))
    if (ClassDiff > 0): myFOM = 1.-((1.-myFOM)**(1/(2*abs(ClassDiff))))
    FOM = CPFOM
    if ((FOM < 0.3) or (FOM > 0.7)): FOM = myFOM
    fom_vec.append(FOM)

#---------------------  NOW CALCULATE BASED ON SEED -- signed (SEED1 - Seed2)           ------------------------
for j in range(0,30):
  for i in range(1101,1465):
    tourney.loc[ (tourney.Team1 == i) & (tourney.Season == year-j), 'Team1Seed']  = seeds[ (seeds.Season == year-j) & (seeds.Team == i)]['Seed'].mean()
    tourney.loc[ (tourney.Team2 == i) & (tourney.Season == year-j), 'Team2Seed']  = seeds[ (seeds.Season == year-j) & (seeds.Team == i)]['Seed'].mean()
tourney['SeedDiff'] = tourney['Team1Seed'] - tourney['Team2Seed']

#---------------------  NOW CALCULATE FREE THROW SCORE FOR EACH TEAM           ------------------------
for i in range(1101, 1465):
  tourney.loc[ (tourney.Team1 == i), 'Team1FTP'] = teams[ (teams.Team_Id == i)]['FTP'].mean()
  tourney.loc[ (tourney.Team2 == i), 'Team2FTP'] = teams[ (teams.Team_Id == i)]['FTP'].mean()
  tourney.loc[ (tourney.Team1 == i), 'Team1NF']  = teams[ (teams.Team_Id == i)]['nFouls'].mean()
  tourney.loc[ (tourney.Team2 == i), 'Team2NF']  = teams[ (teams.Team_Id == i)]['nFouls'].mean()
tourney['FTScore'] = tourney.Team1FTP * tourney.Team2NF - tourney.Team2FTP * tourney.Team1NF

#And the opponent score
for i in range(1101, 1465):
  tourney.loc[ (tourney.Team1 == i), 'Team1OPFOM1']  = teams[ (teams.Team_Id == i)]['OPFOM1'].mean()
  tourney.loc[ (tourney.Team2 == i), 'Team2OPFOM1']  = teams[ (teams.Team_Id == i)]['OPFOM1'].mean()
tourney['OPFOMDIF'] = tourney.Team1OPFOM1 - tourney.Team2OPFOM1

#---------------------  NOW CALCULATE OFF AND DEF SCORE FOR EACH TEAM           ------------------------
#Start by getting the variables
for i in range(1101, 1465):
  tourney.loc[ (tourney.Team1 == i), 'TO1'] = teams[ (teams.Team_Id == i)]['NTO'].mean()
  tourney.loc[ (tourney.Team2 == i), 'TO2'] = teams[ (teams.Team_Id == i)]['NTO'].mean()
  tourney.loc[ (tourney.Team1 == i), 'ST1'] = teams[ (teams.Team_Id == i)]['NST'].mean()
  tourney.loc[ (tourney.Team2 == i), 'ST2'] = teams[ (teams.Team_Id == i)]['NST'].mean()
  tourney.loc[ (tourney.Team1 == i), 'AST1'] = teams[ (teams.Team_Id == i)]['NAST'].mean()
  tourney.loc[ (tourney.Team2 == i), 'AST2'] = teams[ (teams.Team_Id == i)]['NAST'].mean()
  tourney.loc[ (tourney.Team1 == i), 'BLK1'] = teams[ (teams.Team_Id == i)]['NBLK'].mean()
  tourney.loc[ (tourney.Team2 == i), 'BLK2'] = teams[ (teams.Team_Id == i)]['NBLK'].mean()
  tourney.loc[ (tourney.Team1 == i), 'OR1'] = teams[ (teams.Team_Id == i)]['NOR'].mean()
  tourney.loc[ (tourney.Team2 == i), 'OR2'] = teams[ (teams.Team_Id == i)]['NOR'].mean()
  tourney.loc[ (tourney.Team1 == i), 'DR1'] = teams[ (teams.Team_Id == i)]['NDR'].mean()
  tourney.loc[ (tourney.Team2 == i), 'DR2'] = teams[ (teams.Team_Id == i)]['NDR'].mean()
  tourney.loc[ (tourney.Team1 == i), 'FGP1'] = teams[ (teams.Team_Id == i)]['FGP'].mean()
  tourney.loc[ (tourney.Team2 == i), 'FGP2'] = teams[ (teams.Team_Id == i)]['FGP'].mean()
  tourney.loc[ (tourney.Team1 == i), 'FGA1'] = teams[ (teams.Team_Id == i)]['FGA'].mean()
  tourney.loc[ (tourney.Team2 == i), 'FGA2'] = teams[ (teams.Team_Id == i)]['FGA'].mean()
  tourney.loc[ (tourney.Team1 == i), 'FGP31'] = teams[ (teams.Team_Id == i)]['FGP3'].mean()
  tourney.loc[ (tourney.Team2 == i), 'FGP32'] = teams[ (teams.Team_Id == i)]['FGP3'].mean()
  tourney.loc[ (tourney.Team1 == i), 'FGA31'] = teams[ (teams.Team_Id == i)]['FGA3'].mean()
  tourney.loc[ (tourney.Team2 == i), 'FGA32'] = teams[ (teams.Team_Id == i)]['FGA3'].mean()
tourney['TO_DIFF'] = tourney.TO1 - tourney.TO2
tourney['ST_DIFF'] = tourney.ST1 - tourney.ST2
tourney['AST_DIFF'] = tourney.AST1 - tourney.AST2
tourney['BLK_DIFF'] = tourney.BLK1 - tourney.BLK2
tourney['OR_DIFF'] = tourney.OR1 - tourney.OR2
tourney['DR_DIFF'] = tourney.DR1 - tourney.DR2
tourney['FGP_DIFF'] = tourney.FGP1 - tourney.FGP2
tourney['FGA_DIFF'] = tourney.FGA1 - tourney.FGA2
tourney['FGP3_DIFF'] = tourney.FGP31 - tourney.FGP32
tourney['FGA3_DIFF'] = tourney.FGA31 - tourney.FGA32
#Then train the offensive score
train = df[(df.Season >= 2012) & (df.Season <= 2015)][['FOM1','TO_DIFF','AST_DIFF','ST_DIFF','OR_DIFF','FGP_DIFF','FGA_DIFF','FGP3_DIFF','FGA3_DIFF']]
train_data = train.values
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data[0::,1::],train_data[0::,0].astype(str))
test = tourney[(tourney.Season >= 2001) & (tourney.Season <= 2015)][['FOM1','TO_DIFF','AST_DIFF','ST_DIFF','OR_DIFF','FGP_DIFF','FGA_DIFF','FGP3_DIFF','FGA3_DIFF']]
test_data = test.values
output = forest.predict(test_data[0::,1::]).astype(float)
tourney.loc[ (tourney.Season >= 2001) & (tourney.Season <= 2015), 'OFFSCORE'] = output
offscore_vec = []
for i in range(1101,1465):
  for j in range(i+1, 1465): 
    NTO1  = teams[ (teams.Team_Id == i)]['NTO'].mean()
    NTO2  = teams[ (teams.Team_Id == j)]['NTO'].mean()
    NST1  = teams[ (teams.Team_Id == i)]['NST'].mean()
    NST2  = teams[ (teams.Team_Id == j)]['NST'].mean()
    NAST1 = teams[ (teams.Team_Id == i)]['NAST'].mean()
    NAST2 = teams[ (teams.Team_Id == j)]['NAST'].mean()
    NOR1  = teams[ (teams.Team_Id == i)]['NOR'].mean()
    NOR2  = teams[ (teams.Team_Id == j)]['NOR'].mean()
    FGP1  = teams[ (teams.Team_Id == i)]['FGP'].mean()
    FGP2  = teams[ (teams.Team_Id == j)]['FGP'].mean()
    FGA1  = teams[ (teams.Team_Id == i)]['FGA'].mean()
    FGA2  = teams[ (teams.Team_Id == j)]['FGA'].mean()
    FGP31 = teams[ (teams.Team_Id == i)]['FGP3'].mean()
    FGP32 = teams[ (teams.Team_Id == j)]['FGP3'].mean()
    FGA31 = teams[ (teams.Team_Id == i)]['FGA3'].mean()
    FGA32 = teams[ (teams.Team_Id == j)]['FGA3'].mean()
    if ((NTO1 != NTO1) or (NTO2 != NTO2)):
      offscore_vec.append(0)
      continue
    OFFSCORE = forest.predict([NTO1-NTO2, NAST1-NAST2, NST1-NST2, NOR1-NOR2, FGP1-FGP2, FGA1-FGA2, FGP31-FGP32, FGA31-FGA32])
    offscore_vec.append(OFFSCORE)
#Then train the defensive score
train = df[(df.Season >= 2012) & (df.Season <= 2015)][['FOM1','TO_DIFF','ST_DIFF','DR_DIFF','BLK_DIFF']]
train_data = train.values
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data[0::,1::],train_data[0::,0].astype(str))
test = tourney[(tourney.Season >= 2001) & (tourney.Season <= 2015)][['FOM1','TO_DIFF','ST_DIFF','DR_DIFF','BLK_DIFF']]
test_data = test.values
output = forest.predict(test_data[0::,1::]).astype(float)
tourney.loc[(tourney.Season >= 2001) & (tourney.Season <= 2015), 'DEFSCORE'] = output
defscore_vec = []
for i in range(1101,1465):
  for j in range(i+1, 1465): 
    NTO1  = teams[ (teams.Team_Id == i)]['NTO'].mean()
    NTO2  = teams[ (teams.Team_Id == j)]['NTO'].mean()
    NST1  = teams[ (teams.Team_Id == i)]['NST'].mean()
    NST2  = teams[ (teams.Team_Id == j)]['NST'].mean()
    NBLK1 = teams[ (teams.Team_Id == i)]['NBLK'].mean()
    NBLK2 = teams[ (teams.Team_Id == j)]['NBLK'].mean()
    NDR1  = teams[ (teams.Team_Id == i)]['NDR'].mean()
    NDR2  = teams[ (teams.Team_Id == j)]['NDR'].mean()
    if ((NTO1 != NTO1) or (NTO2 != NTO2)):
      defscore_vec.append(0)
      continue
    DEFSCORE = forest.predict([NTO1-NTO2, NST1-NST2, NDR1-NDR2, NBLK1-NBLK2])
    defscore_vec.append(DEFSCORE)
#print tourney[((tourney.Season > 2001) & (tourney.Season < 2012))][['OFFSCORE','DEFSCORE','PFOM','SeedDiff','FTScore','ClassDiff','OPFOMDIF']]

###          NOW FOR THE BIG PREDICTIONS!!!!!!!!     #######
#Train on the 2001-2011 tourney data
print "here"
train = tourney[((tourney.Season >= 2001) & (tourney.Season < 2012))][['FOM1','OFFSCORE','DEFSCORE','PFOM','SeedDiff','FTScore','ClassDiff','OPFOMDIF']]
train_data = train.values
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data[0::,1::],train_data[0::,0].astype(str))

#Open file to store predictions
print "here2"
train = tourney[((tourney.Season >= 2001) & (tourney.Season < 2012))][['FOM1','OFFSCORE','DEFSCORE','PFOM','SeedDiff','FTScore','ClassDiff','OPFOMDIF']]
predictions_file = open("round1pred.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["id","pred"])

#Test on the 2012-2015 tourney data
k = -1
for i in range(1101,1465):
  for j in range(i+1, 1465): 
    k += 1
    #Load in variables
    Team1_Class  = teams[ (teams.Team_Id == i)]['Team_Class'].mean()
    Team2_Class  = teams[ (teams.Team_Id == j)]['Team_Class'].mean()
    Team1FTP     = teams[ (teams.Team_Id == i)]['FTP'].mean()
    Team2FTP     = teams[ (teams.Team_Id == j)]['FTP'].mean()
    Team1NF      = teams[ (teams.Team_Id == i)]['NF'].mean()
    Team2NF      = teams[ (teams.Team_Id == j)]['NF'].mean()
    #Predict high-level variables
    ClassDiff = Team1_Class-Team2_Class
    DEFSCORE = offscore_vec[k]
    DEFSCORE = defscore_vec[k]
    FOM      = fom_vec[k]
    OPFOM1   = teams[ (teams.Team_Id == i)]['OPFOM1'].mean()
    OPFOM2   = teams[ (teams.Team_Id == j)]['OPFOM1'].mean()
    FTSCORE  = Team1FTP * Team2NF - Team2FTP * Team1NF
    #Final prediction
    if ((Team1NF != Team1NF) or (Team2NF != Team2NF)):
      output = 0.5    
    else:
      output = forest.predict([OFFSCORE, DEFSCORE, FOM, 0, FTScore, classDiff, OPFOM1-OPFOM2]) 
    #forest.predict(OFFSCORE, DEFSCORE, FOM, 'SeedDiff','FTScore', classDiff,'OPFOMDIF') 
    open_file_object.writerows(zip(i,"_", j, "_", output))

#Close file
predictions_file.close()
