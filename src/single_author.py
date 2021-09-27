import numpy as np
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from tqdm import tqdm

from src.utils import label

features = ["feature1", "feature2", "followers", 
            "is_reply", "is_retweet", "contains_video", "contains_image",
            "word_count", "shared_url_count", "timestamp"]
def train_single_author(dataset, grouped, model_type):
    answers = np.zeros(len(dataset.author))
    for i, author in tqdm(enumerate(dataset.author)):
        try:
            df = grouped.get_group(author)

            # Get features
            train_data = df[features]
            test_data = dataset[dataset["author"] == author][features]

            # Normalization
            scaler = StandardScaler()
            X = scaler.fit_transform(train_data)
            Y = df[label]
            test_data = scaler.transform(test_data)

            # Train and predict
            lr_model = model_type()
            lr_model.fit(X, Y)
            answers[i] = lr_model.predict(test_data)[0]
            if answers[i] > 1300000:
                answers[i] = 1300000
        except:
            answers[i] = np.nan
    return answers * (answers >= 0)

def train_single_author_LGBM(dataset, grouped):
    answers = np.zeros(len(dataset.author))
    for i, author in tqdm(enumerate(dataset.author)):
        try:
            df = grouped.get_group(author)

            # Get features
            train_data = df[features]
            test_data = dataset[dataset["author"] == author][features]

            # Normalization
            scaler = StandardScaler()
            X = scaler.fit_transform(train_data)
            Y = df[label]
            test_data = scaler.transform(test_data)

            # Train and predict
            lr_model = LGBMRegressor(num_leaves=8, max_depth=3, learning_rate=0.07, objective="mae", random_state=0)
            lr_model.fit(X, Y)
            answers[i] = lr_model.predict(test_data)[0]
            if answers[i] > 1300000:
                answers[i] = 1300000
        except:
            answers[i] = np.nan
    return answers * (answers >= 0)

def replace_nan(answer, value):
    return np.nan_to_num(answer, copy=True, nan=value)