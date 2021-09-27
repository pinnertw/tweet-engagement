from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils import train, author_data, test, label, fill

# Features engineering
features = ["feature1", "feature2", "followers", "word_count", 
            "is_reply", "is_retweet", "contains_video", "contains_image",
            "shared_url_count", "timestamp"
           ] + [
    "V{}".format(i) for i in range(1, 1025)
] + ["engagement_"]

features = ['timestamp', 'followers', 'shared_url_count', 'is_reply', 'V8', 'V258',
       'V266', 'V456', 'V479', 'V582', 'V632', 'V687', 'V716', 'V915'] + ["engagement_"]

# Add average-engagement feature
# Train
dataset = train.join(author_data.groupby("author").agg(
    {"engagement" : "mean"}
).fillna(fill).rename(columns={"engagement" : "engagement_"}), on="author")[["engagement"] + features].fillna(fill)

# Test
test = test.join(author_data.groupby("author").agg(
    {"engagement" : "mean"}
).fillna(fill).rename(columns={"engagement" : "engagement_"}), on="author")[features].fillna(fill)

# Train set and label set
train_set = dataset[features]
label_set = dataset["engagement"]
test_data = test[features]

# Data preprocessing
# Seperation train/test
X_train, X_test, Y_train, Y_test= train_test_split(train_set, 
                                                   label_set, 
                                                   train_size=0.75,
                                                   random_state=0,
                                                  )

# PCA
'''
pca = PCA(n_components = 3, random_state=0)
X_train = pca.fit_transform(X_train)
pca.explained_variance_ratio_
X_test = pca.transform(X_test)
test_data = pca.transform(test_data)
train_set = pca.transform(train_set)
'''

# Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
test_data = scaler.transform(test_data)


train_set = scaler.transform(train_set)