import xgboost as xgb
from lib1763191 import *
import sys


# Load the file
print("Loading files...", end="")
training_filename, test_filename = sys.argv[1:]
train = pd.read_csv(training_filename, index_col=0)
test = pd.read_csv(test_filename, index_col=0)
print("   [DONE]")

# Separate the train prices
train_prices = train['SalePrice'].values

# Save the sizes of each set
train_length = train.shape[0]
test_length = test.shape[0]

# Put both dataframes together to clean them and feature engineer
data = pd.concat([train.drop(["SalePrice"], axis=1), test])

# Clean
data = clean(data)

# Add features
data = add_features(data)

# Standardize dataframe
# This is commented because the results are better without it
#data = zscore(data)

# Split them
print("Creating the model...", end="", flush=True)
train_features = data.iloc[0:train_length].values
test_features = data.iloc[train_length::].values

# Predict
regr = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.01,
                 max_depth=6, # 6
                 min_child_weight=1.5,
                 n_estimators=7600,
                 reg_alpha=0.9,
                 reg_lambda=0,
                 subsample=0.2,
                 seed=42,
                 silent=1,
                 objective="reg:linear")

# Create model
regr.fit(train_features, train_prices)
print("   [DONE]")

# Run the model in the test
print("Predicting the prices of the test set...", end="", flush=True)
test_prices = regr.predict(test_features)
print("   [DONE]")

# Output, with the inverse lop1p of the price
export(test.index, test_prices)

# Check performance of the model
# uncomment next line to run the cross validation
#cv(regr, train_features, train_prices, kf=5)
