# -*- coding: utf-8 -*-
"""
Created on Fri May  3 09:30:21 2019

@author: Harshit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV



data = pd.read_csv('./train.csv')
seaborn_data = data.copy()
target = data.amount_spent_per_room_night_scaled
data.drop(['amount_spent_per_room_night_scaled'],axis=1,inplace=True)

test_data = pd.read_csv('./test.csv')
reservation_id = test_data.reservation_id
test_data.drop(['reservation_id'],axis=1,inplace=True)

data.drop(['reservation_id'],axis=1,inplace=True)

train_size = data.shape[0]

data = pd.concat([data,test_data])

data.drop(['memberid'],axis=1,inplace=True)

print(data.shape)

booking_date = data.booking_date
checkin_date = data.checkin_date
checkout_date = data.checkout_date

def convert_to_day(date):
    date,month,year = map(int,date.split('/'))
    return year*10000 + month*30 + date

def convert_to_dayofweek(date):
    
    day, month, year = (int(x) for x in date.split('/'))  
    year = 2000+year
    ans = datetime.date(year, month, day)
    return ans.strftime("%A")
    
day_of_week = checkin_date.apply(convert_to_dayofweek)
booking_date = booking_date.apply(convert_to_day)
checkin_date = checkin_date.apply(convert_to_day)
checkout_date = checkout_date.apply(convert_to_day)




def is_weekend(num):
    if num>=6:
        return 1
    else:
        return 0

def day_to_num(day):
    switcher = {'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
    return int(switcher.get(day))
        
def close_to_christmas(day):
    return np.abs(day-395)

def close_to_summer_vacations(day):
    if day<=180 and day<=240:
        return 0
    elif day<=180:
        return 180-day
    else:
        return 240-day
    
def is_month_end(num):
    if num < 0:
        return 1
    else:
        return 0

def change_type(num):
    return str(num)


data['freq_checkin'] = data.groupby('checkin_date')['checkin_date'].transform('count')
data['freq_checkout'] = data.groupby('checkout_date')['checkout_date'].transform('count')

data['checkin_year'] = np.floor(checkin_date/10000)
data['checkin_month'] = np.floor((checkin_date%400)/30)

data['checkout_year'] = np.floor(checkout_date/10000)
data['checkout_month'] = np.floor((checkout_date%400)/30)

#data['is_new_year'] = pd.DataFrame(data['checkout_year'].values - data['checkin_year'].values)

data['is_month_end'] = list(map(is_month_end,data['checkout_month'].values - data['checkin_month'].values))
checkin_booking_duration = (checkin_date.values - booking_date.values)%400
data['booking_date'] = booking_date % 400
data['checkin_date'] = checkin_date % 400
data['checkout_date'] = checkout_date % 400

data['checkin_booking_duration'] = pd.DataFrame(checkin_booking_duration)

data['day_of_week'] = day_of_week
temp = data.day_of_week.apply(day_to_num)
#data['is_weekend'] = list(map(is_weekend,temp + data.roomnights))
data['day_of_week'] = pd.factorize(data.day_of_week)[0]
data['resort_id'] = pd.factorize(data.resort_id)[0]
data['member_age_buckets'] = pd.factorize(data.member_age_buckets)[0]
data['cluster_code'] = pd.factorize(data.cluster_code)[0]
data['reservationstatusid_code'] = pd.factorize(data.reservationstatusid_code)[0]
data['close_to_christmas'] = data.checkin_date.apply(close_to_christmas)
data['close_to_summer_vacations'] = data.checkin_date.apply(close_to_summer_vacations)
data['total_days'] = pd.DataFrame(data.checkin_booking_duration.values + data.roomnights.values)


data = data.replace(-45,0)
data.season_holidayed_code = data.season_holidayed_code.fillna(0)
data.state_code_residence = data.state_code_residence.fillna(0)


data['freq_resort'] = data.groupby('resort_id')['resort_id'].transform('count')
#data['freq_cluster'] = data.groupby('cluster_code')['cluster_code'].transform('count')
data['freq_main_product_code'] = data.groupby('main_product_code')['main_product_code'].transform('count')
data['freq_persontravellingid'] = data.groupby('persontravellingid')['persontravellingid'].transform('count')
data['freq_resort_type_code'] = data.groupby('resort_type_code')['resort_type_code'].transform('count')
data['freq_member_age_buckets'] = data.groupby('member_age_buckets')['member_age_buckets'].transform('count')
data['freq_day_of_week'] = data.groupby('day_of_week')['day_of_week'].transform('count')



data.drop(['persontravellingid','booking_type_code','reservationstatusid_code'],axis=1,inplace=True)
data.drop(['booking_date','checkin_date','checkout_date','is_month_end'],axis=1,inplace=True)
data.drop(['main_product_code','resort_region_code','freq_main_product_code'],axis=1,inplace=True)

'''data_reg = data[['freq_checkin','freq_checkout','checkin_month','checkout_month','checkin_booking_duration','day_of_week',
                 'freq_resort','close_to_christmas','close_to_summer_vacations','total_days','freq_main_product_code',
                 'freq_resort_type_code','freq_member_age_buckets','freq_day_of_week','numberofadults',
                 'numberofchildren','roomnights','total_pax','freq_persontravellingid','freq_cluster']]

data_reg['checkin_month'] = pd.Categorical(data_reg['checkin_month'])
oh_checkin_month = pd.get_dummies(data_reg['checkin_month'],prefix='one_hot_')
data_reg.drop(['checkin_month'],axis=1,inplace=True)
data_reg = pd.concat([data_reg,oh_checkin_month],axis=1)

data_reg['checkout_month'] = pd.Categorical(data_reg['checkout_month'])
oh_checkin_month = pd.get_dummies(data_reg['checkout_month'],prefix='one_hot_')
data_reg.drop(['checkout_month'],axis=1,inplace=True)
data_reg = pd.concat([data_reg,oh_checkin_month],axis=1)

data_reg['day_of_week'] = pd.Categorical(data_reg['day_of_week'])
oh_checkin_month = pd.get_dummies(data_reg['day_of_week'],prefix='one_hot_')
data_reg.drop(['day_of_week'],axis=1,inplace=True)
data_reg = pd.concat([data_reg,oh_checkin_month],axis=1)'''





train_data = data.iloc[:train_size,:]
test_data = data.iloc[train_size:,:]

seaborn_data = pd.concat([train_data,target],axis=1)

train_data = train_data.values
target = target.values
test_data = test_data.values

l = list(data.columns)
for i,name in enumerate(l):
    print(i,name)
    
    
'''train_data_reg = data_reg.iloc[:train_size,:]
test_data_reg = data_reg.iloc[train_size:,:]

train_data_reg = train_data_reg.values
test_data_reg = test_data_reg.values'''


'''import seaborn as sns
sns.lmplot(x='freq_checkin',y='amount_spent_per_room_night_scaled',data = seaborn_data);
plt.show()'''
'''from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import AdaBoostRegressor

xgb_1 = xgb.XGBRegressor(objective ='reg:linear',colsample_bytree = 0.4,learning_rate = 0.1,max_depth = 8,
                          alpha = 4,n_estimators = 150, seed=123)
gbm = xgb.XGBRegressor(objective ='reg:linear',colsample_bytree = 0.4,learning_rate = 0.1,max_depth = 8,
                          alpha = 1,n_estimators = 150, seed=123)
rf = xgb.XGBRegressor(objective ='reg:linear',colsample_bytree = 0.4,learning_rate = 0.05,max_depth = 10,
                          alpha = 0,n_estimators = 150, seed=123)
xgb_2 = xgb.XGBRegressor(objective ='reg:linear',colsample_bytree = 0.3,learning_rate = 0.1,max_depth = 8,
                          alpha = 4,n_estimators = 150, seed=123)
xgb_3 = xg_reg = xgb.XGBRegressor(objective ='reg:linear',colsample_bytree = 0.4,learning_rate = 0.05,max_depth = 10,
                          alpha = 1,n_estimators = 150, seed=123)


meta_model = ElasticNet(random_state=0)

train_X, val_X, train_Y, val_Y = train_test_split(train_data,target,test_size = 0.15)

xgb_1.fit(train_X,train_Y)
print('xgb done!!')
gbm.fit(train_X,train_Y)
print('gbm done!!')
#ada.fit(train_X,train_Y)
print('ada done!!')
rf.fit(train_X,train_Y)
print('rf done!!')
xgb_2.fit(train_X,train_Y)
print('xgb2 done!!')
xgb_3.fit(train_X,train_Y)
print('xgb2 done!!')

print(np.sqrt(mean_squared_error(train_Y,xgb_1.predict(train_X))))
print(np.sqrt(mean_squared_error(train_Y,gbm.predict(train_X))))
print(np.sqrt(mean_squared_error(train_Y,rf.predict(train_X))))
print(np.sqrt(mean_squared_error(train_Y,xgb_2.predict(train_X))))
print(np.sqrt(mean_squared_error(train_Y,xgb_3.predict(train_X))))


xgb_1_val = xgb_1.predict(val_X)
gbm_val = gbm.predict(val_X)
#ada_val = ada.predict(val_X)
rf_val = rf.predict(val_X)
xgb_2_val = xgb_2.predict(val_X)
xgb_3_val = xgb_3.predict(val_X)

print('val done')

val_X = np.column_stack((xgb_1_val,gbm_val,rf_val,xgb_2_val,xgb_3_val))
meta_model.fit(val_X,val_Y)
print('meta model done!!')

preds = meta_model.predict(val_X)
rmse = np.sqrt(mean_squared_error(val_Y,preds))
print('RMSE : %f' % rmse)

xgb_1_test = xgb_1.predict(test_data)
gbm_test = gbm.predict(test_data)
#ada_test = ada.predict(test_data)
rf_test = rf.predict(test_data)
xgb_2_test = xgb_2.predict(test_data)
xgb_3_test = xgb_3.predict(test_data)


test_data = np.column_stack((xgb_1_test,gbm_test,rf_test,xgb_2_test,xgb_3_test))
test_preds = meta_model.predict(test_data)

result = pd.DataFrame(data=list(zip(reservation_id,test_preds)),columns=['reservation_id','amount_spent_per_room_night_scaled'])
result.to_csv('output.csv',index=False)'''


from sklearn.model_selection import train_test_split
train_X,val_X,train_Y,val_Y = train_test_split(train_data,target,test_size=0.2)

xg_reg = xgb.XGBRegressor(objective ='reg:linear',colsample_bytree = 0.4,learning_rate = 0.15,max_depth = 10,
                          alpha = 4,n_estimators = 150, seed=1)
xg_reg.fit(train_X,train_Y)

rmse = np.sqrt(mean_squared_error(val_Y, xg_reg.predict(val_X)))
print("RMSE val: %f" % (100*rmse))

xg_reg.fit(train_data,target)

preds = xg_reg.predict(train_data)

rmse = np.sqrt(mean_squared_error(target, preds))
print("RMSE: %f" % (100*rmse))

test_preds = xg_reg.predict(test_data)

result = pd.DataFrame(data=list(zip(reservation_id,test_preds)),columns=['reservation_id','amount_spent_per_room_night_scaled'])
result.to_csv('output.csv',index=False)

xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()


'''from sklearn.ensemble import GradientBoostingRegressor

gbm = GradientBoostingRegressor(learning_rate = 0.1,random_state = 123,n_estimators=150,max_depth=8)

gbm.fit(train_data,target)

preds = gbm.predict(train_data)

rmse = np.sqrt(mean_squared_error(target, preds))
print("RMSE: %f" % (100*rmse))

test_preds = gbm.predict(test_data)

result = pd.DataFrame(data=list(zip(reservation_id,test_preds)),columns=['reservation_id','amount_spent_per_room_night_scaled'])
result.to_csv('output.csv',index=False)'''

'''from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
sc = StandardScaler()
train_data = sc.fit_transform(train_data_reg)
test_data = sc.fit_transform(test_data_reg)

train_X,val_X,train_Y,val_Y = train_test_split(train_data,target,test_size=0.2)

train_data = torch.Tensor(train_data).cuda()
test_data = torch.Tensor(test_data).cuda()
target = torch.Tensor(target).cuda()

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.fc1 = nn.Linear(28,100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100,50)
        self.bn2= nn.BatchNorm1d(50)
        self.out = nn.Linear(100,1)
        
        
        
    def forward(self,X):
        X = torch.relu(self.bn1(self.fc1(X)))
        #X = torch.relu(self.bn2(self.fc2(X)))
        out = torch.relu(self.out(X))
        return out

model = Model().cuda()
loss = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=0.0005)
epoch = 2000


for i in range(epoch):
    optimizer.zero_grad()
    
    forward = model(train_data)
    forward = forward.squeeze_()
    l = torch.sqrt(loss(forward,target))
    
    print('Epoch : %f loss : %f' % (i,l.item()))
    l.backward()
    optimizer.step()
    
preds = model(train_data).cpu().detach().numpy()
rmse = np.sqrt(mean_squared_error(target, preds))
print("RMSE: %f" % (100*rmse))

test_preds = model(test_data).cpu().detach().numpy()

result = pd.DataFrame(data=list(zip(reservation_id,test_preds)),columns=['reservation_id','amount_spent_per_room_night_scaled'])
result.to_csv('output.csv',index=False)'''



'''from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split

sc = StandardScaler()
train_data = sc.fit_transform(train_data_reg)
test_data = sc.fit_transform(test_data_reg)
train_X,val_X,train_Y,val_Y = train_test_split(train_data,target,test_size=0.2)



en = ElasticNet(random_state=1)
en.fit(train_X,train_Y)

rmse = np.sqrt(mean_squared_error(val_Y, en.predict(val_X)))
print("RMSE val: %f" % (100*rmse))

en.fit(train_data,target)

preds = en.predict(train_data)

rmse = np.sqrt(mean_squared_error(target, preds))
print("RMSE: %f" % (100*rmse))

test_preds = en.predict(test_data)

result = pd.DataFrame(data=list(zip(reservation_id,test_preds)),columns=['reservation_id','amount_spent_per_room_night_scaled'])
result.to_csv('output.csv',index=False)'''

'''def my_custom_loss_func(y_true, y_pred):
        diff = np.sqrt(mean_squared_error(y_true,y_pred))
        return diff

from sklearn.metrics import make_scorer
score = make_scorer(my_custom_loss_func, greater_is_better=False)
# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
param_grid = {
        'colsample_bytree':  [0.3,0.4,0.5],
        'learning_rate' : [0.05,0.1],
        'max_depth' : [6,8,10,12],
        'alpha' : [2,4,6,8],
        'n_estimators' : [100,120,150,200]
        }

grid_search = GridSearchCV(xg_reg, param_grid=param_grid,scoring = score, cv=3,n_jobs=8)
grid_search.fit(train_data, target)

report(grid_search.cv_results_)'''


'''regr = RandomForestRegressor(max_depth=10, random_state=0,n_estimators=100)
regr.fit(train_data,target)

preds = regr.predict(train_data)

rmse = np.sqrt(mean_squared_error(target, preds))
print("RMSE: %f" % (100*rmse))

test_preds = regr.predict(test_data)

result = pd.DataFrame(data=list(zip(reservation_id,test_preds)),columns=['reservation_id','amount_spent_per_room_night_scaled'])
result.to_csv('output.csv',index=False)'''

'''from sklearn.svm import SVR
regr = SVR(gamma='scale', C=1.0, epsilon=0.2)
regr.fit(train_data, target)

preds = regr.predict(train_data)

rmse = np.sqrt(mean_squared_error(target, preds))
print("RMSE: %f" % (100*rmse))

test_preds = regr.predict(test_data)

result = pd.DataFrame(data=list(zip(reservation_id,test_preds)),columns=['reservation_id','amount_spent_per_room_night_scaled'])
result.to_csv('output.csv',index=False)'''












