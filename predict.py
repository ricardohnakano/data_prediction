import sys
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn import cross_validation

from datetime import datetime

os.chdir('/home/ricardo/Desktop/nubank/')

input_data = pd.read_csv(sys.argv[1])
correlation_range = sys.argv[2]
option = sys.argv[3]
if len(sys.argv) == 5:
	second_range = sys.argv[4]
else:
	second_range = 0

def cleaning(data, range, index, range2):
	numeric_data = data._get_numeric_data()

	numeric_data = numeric_data.drop('id', 1)

	correlation = numeric_data.corr()["target"]
	if int(index) == 0:
		for value in correlation:
			#--------------[---------0----------]-----------------
			if value < float(range) and value > -float(range):
				numeric_data = numeric_data.drop(correlation[correlation==value].index[0],1)
	elif int(index) == 1:
		for value in correlation:
			#--------------]---------0----------[-----------------
			if (value > float(range) or value < -float(range)) and value <> 1:
				numeric_data = numeric_data.drop(correlation[correlation==value].index[0],1)
	elif int(index) == 2:
		for value in correlation:
			#-----------[-----]-------0------[-----]---------------
			if (value > float(range) and value < float(range) + float(range2))  or (value < -float(range) and value > -float(range) - float(range2)) and value <> 1:
				numeric_data = numeric_data.drop(correlation[correlation==value].index[0],1)
	elif int(index) == 3:
		for value in correlation:
			#-----------]-----[-------0------]-----[----------------
			if (value > float(range) + float(range2) or (value < float(range) and value > -float(range)) or value < -float(range) - float(range2)) and value <> 1:
				numeric_data = numeric_data.drop(correlation[correlation==value].index[0],1)
	return numeric_data 

def modeling(data):
	Y = data.target.values
	data = data.drop('target', 1)	
	columns = data.columns.tolist()
	target = "target"

	#train = data.sample(frac=0.8, random_state=1)
	#test = data.loc[~data.index.isin(train.index)]
	

	model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=0)
	model.fit(data[columns], Y)
	#predictions = model.predict(data[columns])



	#print 'R squared = ' + str(r2_score(test[target],predictions))
	score_rf =  cross_validation.cross_val_score(model, data, Y, cv=5, scoring='r2', n_jobs=-1)
	print("Random Forest Squared Regression: %0.2f (+/- %0.2f)" % (score_rf.mean(), score_rf.std() * 2))

	return score_rf

def main():
	log = open("log", "a")
	log.write(str(datetime.now()) + "-START" + " (range = " + str(correlation_range) + " index = " + str(option) + " range2 = " +  str(second_range) +  ")\n")

	cleaned_data = cleaning(input_data, correlation_range, option, second_range)

	log.write(str(datetime.now()) + " - \n")

	r2 = modeling(cleaned_data)

	log.write(str(datetime.now()) + "-FINISH - Random Forest Squared Regression: %0.2f (+/- %0.2f)" % (r2.mean(), r2.std() * 2) + "\n\n")
	log.close()
if __name__ == '__main__':
	main()