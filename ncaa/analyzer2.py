import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
import csv

#Change the max rows
pd.set_option('display.width', 1000)

#Load in regular season data
df = pd.read_csv('2015_predictors.csv', header = 0)

#Load in training data
train_df = pd.read_csv('train.csv', header = 0)

#Load in Team Information
teams = pd.read_csv('data/Teams.csv', header = 0)

#Train random forest
train_data = train_df.values
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data[0::,1::],train_data[0::,0].astype(str))

#Predict all 2015 matchups
k = -1
for i in range(1101,1465):
  for j in range(i+1, 1465): 
    test_data = df[0::,1::].values
    output = forest.predict(df).astype(float)
    printMe = "2015_%i_%i,%f         %s,%s" % (i, j, output, Team1_Name, Team2_Name)
    open_file_object.writerows(printMe)
 
    #Final prediction
    #output = 0.5
    #if ((Team1NF != Team1NF) or (Team2NF != Team2NF)):
    #  output = 0.5    
    #else:
    #  a = [OFFSCORE_AG, DEFSCORE_AG, FOM, SEEDDIF, FTSCORE, ClassDiff, OPFOM1-OPFOM2]
    #  print a
    #  output = forest.predict(a).astype(float)
    #printMe = "2015_%i_%i,%f         %s,%s" % (i, j, output, Team1_Name, Team2_Name)
    #open_file_object.writerows(printMe)

#Close file
#predictions_file.close()
