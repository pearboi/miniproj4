import json, hmac, hashlib, time, requests, base64
from requests.auth import AuthBase
from endpoints import Controller
#pip install endpoints
#pip install requests
#pip install robin_stocks
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree 

#read data





class Model(Controller):
    def POST(self, **kwargs):
        cases = kwargs['params']['cases']
        days = kwargs['params']['days']
        print(cases)
        print(days)
        
        x =  getModel(cases, days)
        print(x[0])
        prediction = {'prediction': x[0]}
        return '{}'.format(x[0])

class Coinbasepro(Controller):
    def GET(self):
        return getCoinbasePro()

def getCoinbasePro():
   return 0

def getModel(cases, days):
    covid_deaths = pd.read_csv("miniproj4.csv", header = 0)

    total_new_case = {}
    total_new_death = {}

    covid_deaths['submission_date'] = pd.to_datetime(covid_deaths['submission_date'])
    covid_deaths = covid_deaths.sort_values(by=['state', 'submission_date'])
    dates = covid_deaths['submission_date'].map(pd.Timestamp.date).unique()

    # covid_deaths[covid_deaths['new_case'] < 0] = 0
    # covid_deaths[covid_deaths['new_death'] < 0] = 0

    # covid_deaths['new_case'][covid_deaths['new_case']<0] = 0
    # covid_deaths['new_death'][covid_deaths['new_death']<0] = 0

    for date in dates:
        date_info = covid_deaths[(covid_deaths['submission_date'].map(pd.Timestamp.date) == date)]
        #texas_info = covid_deaths[(covid_deaths['submission_date'].map(pd.Timestamp.date) == date) & (covid_deaths['state'] == 'TX')]
        total_new_case[date] = date_info['new_case'].sum()
        total_new_death[date] = date_info['new_death'].sum()
        #print(total_new_case)
        #print(total_new_death)

    print('clean')
    # data_size = len(dates)
    # cutoff = int(data_size*.7)
    # print(cutoff)
    # new_case = list(total_new_case.values())
    # new_death = list(total_new_death.values())
    # new_death = [int(death) for death in new_death]

    # first = (list(total_new_case.keys())[0])
    # elapsed_dates = list(total_new_case.keys())
    # elapsed_days = [(dates - first).days for dates in elapsed_dates]
    # elapsed_days = np.array(elapsed_days)

    # new_case = np.array(new_case)

    # training_new_case = new_case[:cutoff]
    # test_new_case = new_case [cutoff:]
    # print(training_new_case.size)
    # training_elapsed_days = elapsed_days[:cutoff]
    # test_elapsed_days = elapsed_days[cutoff:]

    # training_new_death = new_death[:cutoff]
    # test_new_death = new_death[cutoff:]

    # training = np.vstack((training_new_case, training_elapsed_days)).T
    # test = np.vstack((test_new_case, test_elapsed_days)).T

    # clf = tree.DecisionTreeClassifier()
    # clf = clf.fit(training, training_new_death)

    # test_data = [[i, i] for i in range(1,1000)]

    # predict = clf.predict(test)
    # # print(predict)
    # # print(test_new_death)
    # diff = predict - test_new_death
    # # print(diff)
    # # print(len(diff))
    # print(diff.mean())

    # plt.scatter(range(len(diff)), diff, s=1)
    # plt.xticks(range(len(diff)), diff)
    # plt.xlabel("Time")
    # plt.ylabel("Cases")
    # plt.title("New Covid 19 Cases over Time")
    # plt.show()
    # plt.clf()

    data_size = len(dates)
    cutoff = int(data_size*.7)

    new_case = list(total_new_case.values())
    new_death = list(total_new_death.values())
    new_death = [int(death) for death in new_death]

    first = (list(total_new_case.keys())[0])
    elapsed_dates = list(total_new_case.keys())
    elapsed_days = [(dates - first).days for dates in elapsed_dates]
    elapsed_days = np.array(elapsed_days)

    new_case = np.array(new_case)

    print('clean2')
    # training_new_case = new_case[:cutoff]
    # test_new_case = new_case [cutoff:]
    # print(training_new_case.size)
    # training_elapsed_days = elapsed_days[:cutoff]
    # test_elapsed_days = elapsed_days[cutoff:]

    # training_new_death = new_death[:cutoff]
    # test_new_death = new_death[cutoff:]

    training = np.vstack((new_case, elapsed_days, new_death)).T
    # training = np.random.choice(training.shape[0], cutoff, replace=False)
    training = training[np.random.choice(training.shape[0], cutoff, replace=False)]
    trainingx = training[:,[0,1]]
    trainingy = training[:,[2]]

    test = np.vstack((new_case, elapsed_days, new_death)).T
    # test = np.random.choice(test, data_size-cutoff)
    test = test[np.random.choice(test.shape[0],data_size-cutoff, replace=False)]
    testx = test[:,[0,1]]
    testy = test[:,[2]]

    print('train data created')

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(trainingx, trainingy)
    print('trained')
    # print(testx)
    # test_data = [[np.random.randint(1, high=100000), i] for i in range(1,1000)]
    # print(test_data)
    # predict = clf.predict(testx)
    predict = clf.predict([[cases, days]])
    print(predict)
    #print(predict)
    # print(predict)
    # print(testy)
    #diff = predict - testy
    # print(diff)
    #print(diff.mean())

    # plt.scatter(range(len(diff[0])), diff[0], s=1)
    # plt.xticks(range(len(diff[0])), diff[0])
    # plt.xlabel("Time")
    # plt.ylabel("Cases")
    # plt.title("New Covid 19 Cases over Time")
    # plt.show()
    # plt.clf()
    return predict

def main():
    #response = requests.get(cbPro_api_url + 'accounts', auth=auth)
    #print(response.json())
    getCoinbasePro()
    getRobinhood()

if __name__ == "__main__":
    # execute only if run as a script
    main()