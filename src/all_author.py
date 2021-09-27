from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.utils import author_data, label, test

features = ["feature1", "feature1", "feature2", "followers", "word_count", 
            "is_reply", "is_retweet", "contains_video", "contains_image",
            "shared_url_count", "timestamp"]

# Seperation train/test
X_train, X_test, Y_train, Y_test= train_test_split(author_data[features], 
                                                   author_data[label], 
                                                   train_size=0.75,
                                                   random_state=0,
                                                  )

# Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
test_data = scaler.transform(test[features])


# Final usage set.
train_set = scaler.transform(author_data[features])
label_set = author_data[label]