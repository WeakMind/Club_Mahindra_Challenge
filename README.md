# Club_Mahindra_Challenge
A predictive model that predicts the expenditure of the guests on food and beverages.

Dataset - https://datahack.analyticsvidhya.com/contest/club-mahindra-dataolympics/

Language Used - Python

Framework Used - Pytorch

Architecture - A stacking technique of ensembling is used consisiting of 4 models on the original data , namely, (XGBoost, GradientBoostingRegressor, Random Forest, Adaboost), and Elastic Net on the meta data, giving the root-mean-squared error of 0.99.
