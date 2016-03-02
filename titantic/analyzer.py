import pandas as pd
import numpy as np
import pylab as P
import matplotlib

#Change the max rows
pd.set_option('display.width', 1000)

df = pd.read_csv('train.csv', header = 0)

#Print just the top part
#print df.head(3)

#What type is df?
#print type(df)

#What type is the data it detected?
#print df.dtypes

#Dump the info
#print df.info()

#More info!!
#print df.describe()

#Print out the first 10 ages
#print df['Age'][0:10]
#print df.Age[0:10]

#Get type of age
#print type(df.Age)

#Get average age (for non-zero values)
#print df.Age.median()

#Look at multiple columns
#print df[ ['Sex', 'Pclass', 'Age'] ][0:10]

#Look at just old people
#print df[ df.Age > 60 ][['Sex','Pclass','Age','Survived']]

#Look at missing age values
#print df[ df.Age.isnull() ][['Sex','Pclass','Age','Survived']]

#Print the number of people in each class
#for j in ("male", "female"):
#  print "%ss" % j
#  for i in range(1,4):
#    print " ", i, len(df[ (df.Sex == j) & (df.Pclass == i) ])

#Plot age
#df.Age.hist()
#P.show()

#Cooler age plot
df.Age.dropna().hist(bins=16, range=(0,80), color = 'r')
P.show()

