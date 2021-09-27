import pandas as pd
import numpy as np
# Define label
label = "engagement"

# Define data paths
data_path = "./data/"
author_data_path = data_path + "authorData.csv"
sample_submission_path = data_path + "sampleSubmission.csv"
train_path = data_path + "train.csv"
test_path = data_path + "test.csv"

# Read files
author_data = pd.read_csv(author_data_path)
sample_submission = pd.read_csv(sample_submission_path)
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# fill_engagement with ?
quantile_50 = author_data.engagement.quantile(0.5)
mean_ = author_data.engagement.mean()
fill = 7

# Features
features_author = author_data.columns

# Bins for seperation
followers, followers_bins = pd.qcut(author_data["followers"], q=100, labels=False, retbins=True)
followers_bins[-1] = np.Inf

def cost(model, data, label_):
    return np.abs(label_ - model.predict(data)).mean()

def test_model(model, X_train, X_test, Y_train, Y_test):
    model.fit(X_train, Y_train)
    cost_train, cost_test = cost(model, X_train, Y_train), cost(model, X_test, Y_test)
    print(cost_train, cost_test)
    return model, cost_train, cost_test

def final_model(model, train_set, label_set):
    model.fit(train_set, label_set)
    return model

# Write into file
# If applied on a general model.
def write_submission_global(model, cost_train, cost_test, test_data, output_name="output"):
    sample_submission["engagement"] = model.predict(test_data)
    sample_submission["engagement"] = sample_submission["engagement"] * (sample_submission["engagement"] >= 0)
    sample_submission.to_csv("output/" + output_name + "_{:.2f}_{:.2f}.csv".format(cost_train, cost_test), 
                             index=False)
    return

# If applied on a localized model.
def write_submission_local(answers, cost, output_name="output"):
    print(cost)
    sample_submission["engagement"] = answers
    sample_submission.to_csv("output/" + output_name + "_{:.2f}.csv".format(cost), 
                             index=False)
    return