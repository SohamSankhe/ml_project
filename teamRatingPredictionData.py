import sqlQueries
import dbConnector
import numpy as np
import pandas as pd


def getRegressionData(matchList):

    con = dbConnector.getConnection()

    teamRatingMatrix = np.zeros((matchList.__len__() * 2, 25))  # replace col with no of features
    #print('np.shape(teamRatingMatrix): ', np.shape(teamRatingMatrix))

    currentRowCounter = 0

    for match in matchList:

        # print('Match_id: ', match)


        # get player ratings and team rating
        ratingsData = pd.read_sql(sqlQueries.PLAYER_TEAM_RATINGS.format(match_identifier=match), con)
        # print('\nratingsData:\n',ratingsData)

        # fill all the null ratings (due to data inconsistencies) with avg ratings
        ratingsData['player_rating'].fillna(ratingsData['player_rating'].mean(), inplace=True)

        # get xTraining : player ratings vector and yTraining: team rating

        # drop extra players
        ind = 0
        for i in range(1, ratingsData.shape[0] - 1):
            if ratingsData['team_id'][i] != ratingsData['team_id'][i - 1]:
                ind = i
                break
        # print('index: ', ind)

        team1Ratings = ratingsData.loc[:10, :]
        team2Ratings = ratingsData.loc[ind:(ind + 10), :]

        # print('team1Ratings: \n', team1Ratings)
        # print('team2Ratings: \n', team2Ratings)

        # getting below feature vectors per match:
        # matchid, team1 id,  team1 player rating * 11, team2 player rating * 11 -> team1 rating
        # matchid, team2 id, team2 player rating * 11, team1 player rating * 11 -> team2 rating

        team1PlyrRatingList = list(team1Ratings['player_rating'].values.flatten())
        team2PlyrRatingList = list(team2Ratings['player_rating'].values.flatten())

        temp1 = team1PlyrRatingList.copy()
        temp2 = team2PlyrRatingList.copy()

        trainingRow1 = []
        trainingRow1.append(match)
        trainingRow1.append(team1Ratings['team_id'][0])
        trainingRow1.extend(temp1)
        trainingRow1.extend(team2PlyrRatingList)

        trainingRow2 = []
        trainingRow2.append(match)
        trainingRow2.append(team2Ratings['team_id'][ind])
        trainingRow2.extend(temp2)
        trainingRow2.extend(team1PlyrRatingList)

        # append response ie team rating to features
        trainingRow1.append(team1Ratings['team_rating'][0])
        trainingRow2.append(team2Ratings['team_rating'][ind])

        # print('trainingRow1:\n', trainingRow1)
        # print('trainingRow2:\n', trainingRow2)

        # add vectors to matrix
        teamRatingMatrix[currentRowCounter, :] = trainingRow1
        currentRowCounter += 1
        teamRatingMatrix[currentRowCounter, :] = trainingRow2
        currentRowCounter += 1
        # end of for

    teamRatingMatrix = np.mat(teamRatingMatrix)

    # np.set_printoptions(precision=4)
    # print('\nteamRatingMatrix:\n', teamRatingMatrix)


    '''
    training = teamRatingMatrix[:100, :]
    testing = teamRatingMatrix[101:, :]
    rows, cols = np.shape(teamRatingMatrix)
    xTraining = training[:, 0:cols - 1]
    yTraining = training[:, cols - 1]
    xTest = testing[:, :cols - 1]
    yTest = testing[:, cols - 1]
    '''

    return teamRatingMatrix


def getRegressionDataTraining(matchList):

    teamRatingMatrix = getRegressionData(matchList)
    rows, cols = np.shape(teamRatingMatrix)
    xTraining = teamRatingMatrix[:, 2:cols - 1]  # remove col for match id, team id and team rating
    yTraining = teamRatingMatrix[:, cols - 1]  # take last column

    return xTraining, yTraining


def getRegressionDataTest(matchList):

    teamRatingMatrix = getRegressionData(matchList)

    rows, cols = np.shape(teamRatingMatrix)
    xTest = teamRatingMatrix[:, 2:cols - 1] # remove col for match id, team id and team rating
    yTest = teamRatingMatrix[:, cols - 1] # take last column

    # get matchid, team1 id, team1 rating
    #     matchid, team2 id, team2 rating

    #testResult = np.zeros(())

    return xTest, yTest, teamRatingMatrix
