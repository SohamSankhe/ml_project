import utils
import dbConnector
import pandas as pd
import numpy as np
import sqlQueries


def getMatchData(matchList, matchQuery):

    con = dbConnector.getConnection()

    # get matchdata for matches
    matchesCs = utils.getCommaSepForm(matchList)
    matchData = pd.read_sql(matchQuery.format(match_identifier=matchesCs), con)

    rowsToIgnore = ['match_id', 'home_team_id', 'away_team_id', 'full_time_score']
    reducedMatchData = matchData.drop(rowsToIgnore, axis=1)

    '''
    print('check')
    print('np.shape(reducedMatchData): ', np.shape(reducedMatchData))
    print('np.shape(matchData): ', np.shape(matchData))
    '''

    return reducedMatchData, matchData


def getMatchDataTraining(trainingMatches):

    reducedMatchData, matchTrainingData = getMatchData(trainingMatches, sqlQueries.MATCH_FEATURES_TRAINING)

    return reducedMatchData


def getMatchDataTesting(testingMatches):

    reducedMatchData, matchTestingData = getMatchData(testingMatches, sqlQueries.MATCH_FEATURES_TESTING)

    return reducedMatchData, matchTestingData


def divideClassificationData(reducedMatchData, classLimit):

    # divide to get training and testing matrix
    rows, columns = np.shape(reducedMatchData)
    classTrainingLimit = rows - int(rows * classLimit)  # 1 percent data for testing
    print('classTrainingLimit: ', classTrainingLimit)

    # last column - match_outcome is response (1 - win, 0 - draw, -1 - loss)
    reducedMatchData = np.mat(reducedMatchData)
    xTrainingClass = reducedMatchData[:classTrainingLimit, 0:columns - 1]  # last column is match outcome
    yTrainingClass = reducedMatchData[:classTrainingLimit, columns - 1]
    xTestClass = reducedMatchData[classTrainingLimit:, 0:columns - 1]
    yTestClass = reducedMatchData[classTrainingLimit:, columns - 1]

    return xTrainingClass, yTrainingClass, xTestClass, yTestClass


def mergeTeamRatingsAndMatchStats(matchTestingData, teamRatingsForTest):

    # merging match data with team ratings predicted earlier
    mTestRowCtr = 0
    while mTestRowCtr < (matchTestingData.shape[0]):
        l1 = matchTestingData['match_id'][mTestRowCtr]
        r1 = matchTestingData['home_team_id'][mTestRowCtr]
        dfLhs = teamRatingsForTest.loc[(teamRatingsForTest['match_id'] == l1) & (teamRatingsForTest['team_id'] == r1)]
        matchTestingData.set_value(mTestRowCtr, 'home_team_rating', dfLhs['predicted_rating'].values[0])

        l2 = matchTestingData['match_id'][mTestRowCtr]
        r2 = matchTestingData['away_team_id'][mTestRowCtr]
        dfRhs = teamRatingsForTest.loc[(teamRatingsForTest['match_id'] == l2) & (teamRatingsForTest['team_id'] == r2)]

        matchTestingData.set_value(mTestRowCtr, 'away_team_rating', dfRhs['predicted_rating'].values[0])
        mTestRowCtr += 1

    pd.set_option('display.expand_frame_repr', False)
    print('\nmatchTestingData: \n', matchTestingData)

    return matchTestingData