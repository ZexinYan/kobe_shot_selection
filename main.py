
# coding: utf-8

# In[21]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


# In[22]:


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


# In[23]:


data.info()


# In[24]:


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


# In[28]:


print(train_data.combined_shot_type.describe())


# In[29]:


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


# In[30]:


print(train_data.action_type.describe())
print(train_data.action_type.unique())


# In[31]:


court_scale, alpha = 5, 0.5
train_data['unique_first_word'] = train_data.action_type.str.split(' ').str[0]
plot_category('unique_first_word')


# In[32]:


def get_acc(feature):
    ct = pd.crosstab(train_data.shot_made_flag, train_data[feature]).apply(lambda x:x/x.sum(), axis=0)
    x, y = ct.columns, ct.values[1, :]
    plt.figure(figsize=(7, 5))
    plt.xlabel(feature)
    plt.ylabel('% shot made')
    plt.plot(x, y)


# In[34]:


train_data.shot_distance.astype('category').describe()


# In[35]:


get_acc('shot_distance')


# In[36]:


def test(data):
    clf = RandomForestClassifier(n_jobs=-1)
    return cross_val_score(clf, data.drop('shot_made_flag', 1), data.shot_made_flag,
                          scoring='roc_auc', cv=10)


# In[37]:


data = train_data[['loc_x', 'loc_y', 'shot_made_flag']]
data = data.dropna()
test(data).mean()


# In[38]:


data = train_data[['loc_y', 'shot_made_flag']]
data = data.dropna()
test(data).mean()


# In[39]:


data = train_data[['shot_distance', 'shot_made_flag']]
data = data.dropna()
test(data).mean()


# In[40]:


get_acc('seconds_remaining')
get_acc('minutes_remaining')


# In[42]:


train_data.season.describe()
train_data.season.unique()
train_data['season_year'] = train_data.season.str.split('-').str[0]

get_acc('season_year')


# In[43]:


data = train_data[['season_year', 'shot_made_flag']].dropna()
test(data).mean()


# In[44]:


action_map = {action: i for i, action in enumerate(train_data.action_type.unique())}
train_data['action_type_enumerated'] = train_data.action_type.map(action_map)
get_acc('action_type_enumerated')


# In[45]:


data = train_data[['action_type_enumerated', 'shot_made_flag']].dropna()
test(data).mean()


# In[46]:


action_map = {action: i for i, action in enumerate(train_data.combined_shot_type.unique())}
train_data['combined_shot_type_enumerate'] = train_data.combined_shot_type.map(action_map)
data = train_data[['combined_shot_type_enumerate', 'action_type_enumerated', 'shot_distance', 'shot_made_flag']].dropna()
test(data).mean()


# In[47]:


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


# In[48]:


sort_encode('combined_shot_type_enumerate')
sort_encode('action_type_enumerated')


# In[49]:


opponent_map = {opp: i for i, opp in enumerate(train_data.opponent.unique())}
train_data['opponent_enumerated'] = train_data.opponent.map(opponent_map)

sort_encode('opponent_enumerated')


# In[53]:


train_data['away'] = train_data.matchup.str.contains('@')
data = train_data[['away', 'shot_made_flag']].dropna()
test(data).mean()


# In[58]:


sort_encode('shot_type')
data = train_data[['sorted_shot_type', 'shot_made_flag']].dropna()
test(data).mean()


# In[59]:


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


# In[60]:


depth, scores = list(range(1, 20, 1)), []
for i in depth:
    clf = RandomForestClassifier(n_jobs=-1, n_estimators=70, max_depth=i, random_state=2016)
    x = cross_val_score(clf, data.drop('shot_made_flag', 1), data.shot_made_flag,
                              scoring='roc_auc', cv=10)
    scores.append(x)
x = [i for i in depth for j in range(10)]
sns.boxplot(x, np.array(scores).flatten())


# In[98]:


model = RandomForestClassifier(n_jobs=-1, n_estimators=70, max_depth=9, random_state=2017)

train = train_data.loc[~train_data.shot_made_flag.isnull(), ['sorted_action_type_enumerated',
                                                            'sorted_shot_type', 'shot_distance', 'sorted_opponent_enumerated', 
                                                             'shot_made_flag', 'away']]

test = train_data.loc[train_data.shot_made_flag.isnull(), ['sorted_action_type_enumerated',
                                                            'sorted_shot_type', 'shot_distance', 'sorted_opponent_enumerated', 
                                                             'shot_id', 'away']]



# In[77]:


train.describe()


# In[99]:


test.describe()
mode = test.sorted_action_type_enumerated.mode()[0]
test.sorted_action_type_enumerated.fillna(mode, inplace=True)


# In[105]:


model.fit(train.drop(['shot_made_flag'], axis=1), train.shot_made_flag)
predict = model.predict_proba(test.drop(['shot_id'], axis=1))


# In[107]:


submission = pd.DataFrame({'shot_id': test.shot_id,
                            'shot_made_flag': predict[:, 1]})
submission.to_csv('submission.csv', index=False)

