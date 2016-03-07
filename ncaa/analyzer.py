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
for j in range(1101, 1465): 
  number = 0
  for i in range(year-3, year):
    nWins = tourney[ (tourney.Season == i) & ((tourney.Wteam == j) | (tourney.Lteam == j))][['Season']].count()
    if (int(nWins) > 1): number += 1
  if (number >= 2): 
    teams.loc[ (teams.Team_Id == j) & (teams.Team_Class == 2), 'Team_Class'] = 1

#------------------------    AFOM FOR EACH TEAM    -------------------------------------
#Calculate the base FOM for the winner and loser in each game
df['FOMW'] = (df.Wscore / (df.Wscore + df.Lscore)) 
numer = (df.Wscore - df.Lscore) - 4
for i in range(len(numer)):
  if (numer[i] < 0): numer[i] = 0
df['FOMW'] += 0.02*numer
df.loc[ (df.FOMW > 0.95), 'FOMW'] = 0.95
df['FOML'] = 1-df.FOMW
for i in range(1101,1465):
  teams.loc[ (teams.Team_Id == i), 'AFOM'] = (df[ (df.Season >= year - 6) & (df.Season != year) & (df.Wteam == i) ]['FOMW'].sum() + df[ (df.Season >= year - 6) & (df.Season != year) & (df.Lteam == i) ]['FOML'].sum())/df[ (df.Season >= year - 6) & (df.Season != year) & ((df.Wteam == i) | (df.Lteam == i))]['FOMW'].count()

#---------------------  NOW CALCULATE PRED FOM FOR EACH REGULAR SEASON GAME ------------------------
df['PFOM'] = -9999
for i in range(1101,1465):
  df.loc[ (df.Wteam == i), 'WteamAFOM']  = teams[ (teams.Team_Id == i)]['AFOM'].mean()
  df.loc[ (df.Lteam == i), 'LteamAFOM']  = teams[ (teams.Team_Id == i)]['AFOM'].mean()
  df.loc[ (df.Wteam == i), 'WteamClass'] = teams[ (teams.Team_Id == i)]['Team_Class'].mean()
  df.loc[ (df.Lteam == i), 'LteamClass'] = teams[ (teams.Team_Id == i)]['Team_Class'].mean()
df['PFOM'] = df.WteamAFOM/(df.WteamAFOM + df.LteamAFOM)

#Once you've "predicted" the winner, need to be about 35% more confident in your prediction
df['blah'] = (df.FOMW - df.WteamAFOM)/df.WteamAFOM
corr = df[ (df.WteamClass == df.LteamClass) & (df.Season >= year - 10) & (df.Season < year) & (df.WteamAFOM > .3) ][['blah']].dropna().mean()
df.loc[ (df.PFOM > 0.55), 'PFOM'] *= 1.+float(corr)
df.loc[ (df.PFOM < 0.45), 'PFOM'] = 1.-(1.-df.PFOM)*(1.+float(corr))

#Finally, need to correct prediction in case schools have different classes
df['ClassDiff'] = (df.WteamClass - df.LteamClass)
df.loc[ (df.ClassDiff < 0), 'PFOM' ] = df.PFOM**(1/(2.0*abs(df.ClassDiff)))
df.loc[ (df.ClassDiff > 0), 'PFOM' ] = 1.-((1.-df.PFOM)**(1/(2.0*abs(df.ClassDiff))))
print df[df.ClassDiff > 0][['Wscore', 'Lscore', 'ClassDiff', 'WteamAFOM', 'LteamAFOM', 'FOMW', 'PFOM']]

#---------------------  NOW CALCULATE BASED ON SEED -- signed (SEED1 - Seed2)^.3 - 0.7  ------------------------

#---------------------  NOW CALCULATE AFOM of previos team 1 - team 2 matchups          ------------------------

#---------------------  NOW CALCULATE HOMEFIELD ADVANTAGE  ----------------------------------------------------


#--------------------  NOW CALCULATE WINNINGNESS PREDICTIONS ----------------------------------------------------












##Origin - cleaned Embarked
#df['Origin'] = df.Embarked.dropna().map( { 'Q': 1, 'S': 2, 'C': 3 } ).astype(int)
#
##Calculate the median age in each class/gender
#median_ages = np.zeros((2,3))
#for i in range(0, 2):
#  for j in range(0, 3):
#    median_ages[i,j] = df[ (df.Gender == i) & (df.Pclass == j+1) ]['Age'].dropna().median()
#
##Make new column for filled age
#df['AgeFill'] = df.Age
##print df[ df.Age.isnull() ][['Gender', 'Pclass', 'Age', 'AgeFill']].head(10)
#for i in range(0, 2):
#  for j in range(0, 3):
#    df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]
##print df[ df.Age.isnull() ][['Gender', 'Pclass', 'Age', 'AgeFill']].head(10)
#df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
#
##Family size variable
#df['FamilySize'] = df['SibSp'] + df['Parch']
#
##Age*Class variable
#df['Age*Class'] = df.AgeFill * df.Pclass
#
##Drop the columns we will not use
#train_ids = df['PassengerId'].values
#df = df.drop( ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age', 'Fare', 'PassengerId'], axis=1) 
#
##Dump the info so we can see if any are non-null
##print df.info()
#
##See which origin is most common
##print "mode: ", df.Origin.mean(), df.Origin.mode()
##for i in range(1,4):
##  print "origin in ", i, " for ", df[ df.Origin == i ]['Origin'].count(), " passengers"
#
##Most common origin is 2, so set that for any if origin is not filled
#df.loc[ df.Origin.isnull(), 'Origin'] = 2 
#
##Define train data
#train_data = df.values
#
##And same thing for the test values
#test_df = pd.read_csv('test.csv', header=0)        # Load the test file into a dataframe
#test_df['Gender'] = test_df.Sex.map( { 'female': 0, 'male': 1} ).astype(int)
#test_df['Origin'] = test_df.Embarked.dropna().map( { 'Q': 1, 'S': 2, 'C': 3 } ).astype(int)
#test_df['AgeFill'] = test_df.Age
#for i in range(0, 2):
#  for j in range(0, 3):
#    test_df.loc[ (test_df.Age.isnull()) & (test_df.Gender == i) & (test_df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]
#test_df['AgeIsNull'] = pd.isnull(test_df.Age).astype(int)
#test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch']
#test_df['Age*Class']  = test_df.AgeFill  * test_df.Pclass
#ids = test_df['PassengerId'].values
#test_df = test_df.drop( ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age', 'Fare', 'PassengerId'], axis=1) 
#test_df.loc[ test_df.Origin.isnull(), 'Origin'] = 2 
#test_data = test_df.values
#
##Run it
#forest = RandomForestClassifier(n_estimators = 100)
#forest = forest.fit(train_data[0::,1::],train_data[0::,0])
##output = forest.predict(test_data).astype(int)
#output = forest.predict(train_data[0::,1::]).astype(int)
#
##Write out values
#predictions_file = open("myfirstforest.csv", "wb")
#open_file_object = csv.writer(predictions_file)
#open_file_object.writerow(["PassengerId","Survived"])
#open_file_object.writerows(zip(train_ids, output, df.Survived, output == df.Survived))
#predictions_file.close()
#
##Final Score
#good = 0
#total = 0
#for i in range(df.Survived.count()):
#  if (output[i] == df.Survived[i]): good += 1
#  total += 1
#print float(good)/float(total)
