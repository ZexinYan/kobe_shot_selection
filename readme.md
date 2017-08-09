

```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
```


```python
data = pd.read_csv('data.csv')
# test_data = pd.read_csv('sample_submission.csv')
# test_id = test['shot_id'].values
# test_data = data.loc[test_id - 1]
train_data = data
target = 'shot_made_flag'
deleted_features = []
dummy_features = []
pos_sign = 1.0
neg_sign = 0.0
```


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 30697 entries, 0 to 30696
    Data columns (total 25 columns):
    action_type           30697 non-null object
    combined_shot_type    30697 non-null object
    game_event_id         30697 non-null int64
    game_id               30697 non-null int64
    lat                   30697 non-null float64
    loc_x                 30697 non-null int64
    loc_y                 30697 non-null int64
    lon                   30697 non-null float64
    minutes_remaining     30697 non-null int64
    period                30697 non-null int64
    playoffs              30697 non-null int64
    season                30697 non-null object
    seconds_remaining     30697 non-null int64
    shot_distance         30697 non-null int64
    shot_made_flag        25697 non-null float64
    shot_type             30697 non-null object
    shot_zone_area        30697 non-null object
    shot_zone_basic       30697 non-null object
    shot_zone_range       30697 non-null object
    team_id               30697 non-null int64
    team_name             30697 non-null object
    game_date             30697 non-null object
    matchup               30697 non-null object
    opponent              30697 non-null object
    shot_id               30697 non-null int64
    dtypes: float64(3), int64(11), object(11)
    memory usage: 5.9+ MB



```python
count_scale, alpha = 7, 0.05
plt.figure(figsize=(2 * count_scale, count_scale * (84.0 / 50.0)))

# hit
plt.subplot(121)
value = train_data[train_data.shot_made_flag == 1]
plt.scatter(value.loc_x, value.loc_y, color='green', alpha=alpha)
plt.title('hit')
ax = plt.gca()
ax.set_ylim([-50, 900])

# miss
plt.subplot(122)
value = train_data[train_data.shot_made_flag == 0]
plt.scatter(value.loc_x, value.loc_y, color='red', alpha=alpha)
plt.title('miss')
ax = plt.gca()
ax.set_ylim([-50, 900])
plt.savefig('hit&miss.png')
```

![output_3_0](http://o7d2h0gjo.bkt.clouddn.com/2017-08-10-output_3_0.png)



```python
print(train_data.combined_shot_type.describe())
```

    count         30697
    unique            6
    top       Jump Shot
    freq          23485
    Name: combined_shot_type, dtype: object



```python
def plot_category(feature):    
    groups = train_data.groupby(feature)
    a = len(groups) / 2 + 1
    plt.figure(figsize=(count_scale * 2, a * 1.1 * count_scale * (84.0 / 50.0)))

    alpha = 0.2
    alphas, n = [], float(len(train_data[feature]))
    for index, (name, group) in enumerate(groups):
        plt.subplot(a,2,index + 1)
        plt.grid(True, linestyle = "-.", color = "r", linewidth = "3", alpha=0.05)  
        plt.scatter(group.loc_x, group.loc_y, alpha=0.1, color='lightskyblue')
        plt.title(name)
        plt.xlim([-300, 300])
        plt.ylim([-50, 900])
    plt.savefig(feature + '.png')
plot_category('combined_shot_type')
```

![output_5_0](http://o7d2h0gjo.bkt.clouddn.com/2017-08-10-output_5_0.png)



```python
print(train_data.action_type.describe())
print(train_data.action_type.unique())
```

    count         30697
    unique           57
    top       Jump Shot
    freq          18880
    Name: action_type, dtype: object
    ['Jump Shot' 'Driving Dunk Shot' 'Layup Shot' 'Running Jump Shot'
     'Driving Layup Shot' 'Reverse Layup Shot' 'Reverse Dunk Shot'
     'Slam Dunk Shot' 'Turnaround Jump Shot' 'Tip Shot' 'Running Hook Shot'
     'Alley Oop Dunk Shot' 'Dunk Shot' 'Alley Oop Layup shot'
     'Running Dunk Shot' 'Driving Finger Roll Shot' 'Running Layup Shot'
     'Finger Roll Shot' 'Fadeaway Jump Shot' 'Follow Up Dunk Shot' 'Hook Shot'
     'Turnaround Hook Shot' 'Running Tip Shot' 'Jump Hook Shot'
     'Running Finger Roll Shot' 'Jump Bank Shot' 'Turnaround Finger Roll Shot'
     'Hook Bank Shot' 'Driving Hook Shot' 'Running Reverse Layup Shot'
     'Driving Finger Roll Layup Shot' 'Fadeaway Bank shot' 'Pullup Jump shot'
     'Finger Roll Layup Shot' 'Turnaround Fadeaway shot'
     'Driving Reverse Layup Shot' 'Driving Slam Dunk Shot'
     'Step Back Jump shot' 'Reverse Slam Dunk Shot' 'Turnaround Bank shot'
     'Running Finger Roll Layup Shot' 'Floating Jump shot'
     'Putback Slam Dunk Shot' 'Running Bank shot' 'Driving Bank shot'
     'Putback Layup Shot' 'Driving Jump shot' 'Putback Dunk Shot'
     'Pullup Bank shot' 'Running Slam Dunk Shot' 'Cutting Layup Shot'
     'Driving Floating Jump Shot' 'Running Pull-Up Jump Shot' 'Tip Layup Shot'
     'Driving Floating Bank Jump Shot' 'Turnaround Fadeaway Bank Jump Shot'
     'Cutting Finger Roll Layup Shot']



```python
court_scale, alpha = 5, 0.5
train_data['unique_first_word'] = train_data.action_type.str.split(' ').str[0]
plot_category('unique_first_word')
```


 



```python
def get_acc(feature):
    ct = pd.crosstab(train_data.shot_made_flag, train_data[feature]).apply(lambda x:x/x.sum(), axis=0)
    x, y = ct.columns, ct.values[1, :]
    plt.figure(figsize=(7, 5))
    plt.xlabel(feature)
    plt.ylabel('% shot made')
    plt.plot(x, y)
```


```python
train_data.shot_distance.astype('category').describe()
```




    count     30697
    unique       74
    top           0
    freq       5542
    Name: shot_distance, dtype: int64




```python
get_acc('shot_distance')
```


![output_10_0](http://o7d2h0gjo.bkt.clouddn.com/2017-08-10-output_10_0.png)




```python
def test(data):
    clf = RandomForestClassifier(n_jobs=-1)
    return cross_val_score(clf, data.drop('shot_made_flag', 1), data.shot_made_flag,
                          scoring='roc_auc', cv=10)
```


```python
data = train_data[['loc_x', 'loc_y', 'shot_made_flag']]
data = data.dropna()
test(data).mean()
```




    0.55374200565088283




```python
data = train_data[['loc_y', 'shot_made_flag']]
data = data.dropna()
test(data).mean()
```




    0.58646797754459024




```python
data = train_data[['shot_distance', 'shot_made_flag']]
data = data.dropna()
test(data).mean()
```




    0.60890346319492961




```python
get_acc('seconds_remaining')
get_acc('minutes_remaining')
```
![output_15_0](http://o7d2h0gjo.bkt.clouddn.com/2017-08-10-output_15_0.png)



![output_15_1](http://o7d2h0gjo.bkt.clouddn.com/2017-08-10-output_15_1.png)




```python
train_data.season.describe()
train_data.season.unique()
train_data['season_year'] = train_data.season.str.split('-').str[0]

get_acc('season_year')
```



![output_16_0](http://o7d2h0gjo.bkt.clouddn.com/2017-08-10-output_16_0.png)


```python
data = train_data[['season_year', 'shot_made_flag']].dropna()
test(data).mean()
```




    0.45970231600241329




```python
action_map = {action: i for i, action in enumerate(train_data.action_type.unique())}
train_data['action_type_enumerated'] = train_data.action_type.map(action_map)
get_acc('action_type_enumerated')
```

![output_18_0](http://o7d2h0gjo.bkt.clouddn.com/2017-08-10-output_18_0.png)




```python
data = train_data[['action_type_enumerated', 'shot_made_flag']].dropna()
test(data).mean()
```




    0.67546900601216164




```python
action_map = {action: i for i, action in enumerate(train_data.combined_shot_type.unique())}
train_data['combined_shot_type_enumerate'] = train_data.combined_shot_type.map(action_map)
data = train_data[['combined_shot_type_enumerate', 'action_type_enumerated', 'shot_distance', 'shot_made_flag']].dropna()
test(data).mean()
```




    0.68623002235176689




```python
def sort_encode(feature):
    ct = pd.crosstab(train_data.shot_made_flag, train_data[feature]).apply(lambda x: x/x.sum(), axis=0)
    temp = list(zip(ct.values[1, :], ct.columns))
    temp.sort()
    new_map = {}
    for index, (acc, old_number) in enumerate(temp):
        new_map[old_number] = index
    new_feature = 'sorted_' + feature
    train_data[new_feature] = train_data[feature].map(new_map)
    get_acc(new_feature)
```


```python
sort_encode('combined_shot_type_enumerate')
sort_encode('action_type_enumerated')
```


![png](output_22_0.png)

![output_22_0](http://o7d2h0gjo.bkt.clouddn.com/2017-08-10-output_22_0.png)
![output_22_1](http://o7d2h0gjo.bkt.clouddn.com/2017-08-10-output_22_1.png)


![png](output_22_1.png)



```python
opponent_map = {opp: i for i, opp in enumerate(train_data.opponent.unique())}
train_data['opponent_enumerated'] = train_data.opponent.map(opponent_map)

sort_encode('opponent_enumerated')
```


![png](output_23_0.png)

![output_23_0](http://o7d2h0gjo.bkt.clouddn.com/2017-08-10-output_23_0.png)


```python
train_data['away'] = train_data.matchup.str.contains('@')
data = train_data[['away', 'shot_made_flag']].dropna()
test(data).mean()
```




    0.51013187677448468




```python
sort_encode('shot_type')
data = train_data[['sorted_shot_type', 'shot_made_flag']].dropna()
test(data).mean()
```




    0.5498194916339959


![output_25_1](http://o7d2h0gjo.bkt.clouddn.com/2017-08-10-output_25_1.png)


![png](output_25_1.png)



```python
data = train_data[['sorted_shot_type', 'action_type_enumerated', 'shot_distance',
           'shot_made_flag', 'away']].dropna()

# We see how score improves with estimators.
estimators, scores = list(range(1, 100, 5)), []
for i in estimators:
    clf = RandomForestClassifier(n_jobs=-1, n_estimators=i, random_state=2016)
    x = cross_val_score(clf, data.drop('shot_made_flag', 1), data.shot_made_flag,
                              scoring='roc_auc', cv=10)
    scores.append(x)
x = [i for i in estimators]
sns.boxplot(x, np.array(scores).flatten())
```




    <matplotlib.axes._subplots.AxesSubplot at 0x10fb0ab70>



![output_26_1](http://o7d2h0gjo.bkt.clouddn.com/2017-08-10-output_26_1.png)

![png](output_26_1.png)



```python
depth, scores = list(range(1, 20, 1)), []
for i in depth:
    clf = RandomForestClassifier(n_jobs=-1, n_estimators=70, max_depth=i, random_state=2016)
    x = cross_val_score(clf, data.drop('shot_made_flag', 1), data.shot_made_flag,
                              scoring='roc_auc', cv=10)
    scores.append(x)
x = [i for i in depth for j in range(10)]
sns.boxplot(x, np.array(scores).flatten())
```




    <matplotlib.axes._subplots.AxesSubplot at 0x10fb0ee10>


![output_27_1](http://o7d2h0gjo.bkt.clouddn.com/2017-08-10-output_27_1.png)


![png](output_27_1.png)



```python
model = RandomForestClassifier(n_jobs=-1, n_estimators=70, max_depth=9, random_state=2017)

train = train_data.loc[~train_data.shot_made_flag.isnull(), ['sorted_action_type_enumerated',
                                                            'sorted_shot_type', 'shot_distance', 'sorted_opponent_enumerated', 
                                                             'shot_made_flag', 'away']]

test = train_data.loc[train_data.shot_made_flag.isnull(), ['sorted_action_type_enumerated',
                                                            'sorted_shot_type', 'shot_distance', 'sorted_opponent_enumerated', 
                                                             'shot_id', 'away']]


```


```python
train.describe()
```




```python
test.describe()
mode = test.sorted_action_type_enumerated.mode()[0]
test.sorted_action_type_enumerated.fillna(mode, inplace=True)
```


```python
model.fit(train.drop(['shot_made_flag'], axis=1), train.shot_made_flag)
predict = model.predict_proba(test.drop(['shot_id'], axis=1))
```




    array([ 0.37233781,  0.29885139,  0.7840646 , ...,  0.78495691,
            0.77867893,  0.19115828])




```python
submission = pd.DataFrame({'shot_id': test.shot_id,
                            'shot_made_flag': predict[:, 1]})
submission.to_csv('submission.csv', index=False)
```


