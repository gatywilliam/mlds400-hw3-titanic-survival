# DATA IMPORTS #
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.exceptions import ConvergenceWarning

# BUILT-IN IMPORTS #
import argparse
import os
import sys
import warnings


argparser = argparse.ArgumentParser()
argparser.add_argument("-si", "--simple_imputer", action="store_true", help="If False, attempts to use scikit-learn's IterativeImputer to impute missing values in the Age column, using the SimpleImputer if a version error occurs. If True, defaults to SimpleImputer. Default False.")
#argparser.add_argument("-dl", "--download_data", action="store_true", help="If True, download dataset from the kagglehub API. REQUIRES KAGGLEHUB API KEY. If False, loads data from /data/*.csv files. REQUIRES FILES TO BE DOWNLOADED FROM KAGGLE 'titanic' COMPETITION.")
argparser.add_argument("-ht", "--hide_test_preds", action="store_true", help="If True, prints only the train-test scores does not print out test predictions (and their PassengerId's) at the end of the output. If False, prints everything. Defaults to False.")
args = argparser.parse_args()

# IMPUTER IMPORTS #
#The sklearn IterativeImputer is experimental. Even with specified package versions, this is best done with some future-proofing. 
if not args.simple_imputer:
	try:
		print(f"Attempting to load scikit-learn IterativeImputer...")
		from sklearn.experimental import enable_iterative_imputer
		from sklearn.impute import IterativeImputer, SimpleImputer
		print(f"IterativeImputer loaded successfully.")
	except Exception as e:
		print(f"scikit-learn IterativeImputer import failed. Using SimpleImputer instead.")
		from sklearn.impute import SimpleImputer
else:
	print(f"Defaulting to scikit-learn SimpleImputer.")
	from sklearn.impute import SimpleImputer


def prep_data(raw_df, embarked_encoder_=None, pclass_encoder_=None, age_imputer_=None, scaler_=None, is_simple_age_imputer=False, return_transformers=False, verbose=False):
	use_cols = ["PassengerId", "Sex", "Pclass", "Age", "SibSp", "Parch", "Fare", "Embarked"]
	if "Survived" in raw_df.columns:
		use_cols.append("Survived")
	df = raw_df[use_cols].set_index("PassengerId").copy()
	if verbose:
		print(f"Columns used for model construction: 'Sex', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'.")
	df["Sex"] = df["Sex"].apply(lambda x: 1 if x.lower() == "male" else 0)

	if verbose:
		print("One-hot encoding 'Embarked' feature...")
	if embarked_encoder_:
		embarked_encoder = embarked_encoder_
	else:
		embarked_encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")

	embarked_cols = embarked_encoder.fit_transform(df[["Embarked"]])
	if "Embarked_nan" in embarked_cols.columns:
		embarked_cols = embarked_cols.drop(columns=["Embarked_nan"])
	if verbose:
		print(f"One-hot encoded 'Embarked' columns: {list(embarked_cols.columns)}")
	df = df.drop(columns=["Embarked"]).join(embarked_cols)

	if verbose:
		print("One-hot encoding 'Pclass' ('Passenger class') feature...")
	if pclass_encoder_:
		pclass_encoder = pclass_encoder_
	else:
		pclass_encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
	
	pclass_cols = pclass_encoder.fit_transform(df[["Pclass"]])
	if verbose:
		print(f"One-hot encoded 'Pclass' feature names: {list(pclass_cols.columns)}")
	df = df.drop(columns=["Pclass"]).join(pclass_cols)

	if "Survived" in df.columns:
		X = df.drop(columns=["Survived"])
		y = df["Survived"]
	else:
		X = df
		y = None

	if verbose:
		print(f"Found {X['Age'].isna().sum()} null values in 'Age' column. Imputing empty values...")

	use_simple = True
	if age_imputer_:
		if is_simple_age_imputer:
			X["Age"] = age_imputer_.fit_transform(X[["Age"]])
		else:
			X = age_imputer_.fit_transform(X)
			use_simple = False
	else:
		if not args.simple_imputer:
			try:
				age_imputer = ColumnTransformer(
				# age_imputer = imputer_transformer = ColumnTransformer(
				    transformers=[
				        ('impute_nums', IterativeImputer(random_state=93), ["Age"])
				    ],
				    remainder='passthrough'
				).set_output(transform="pandas")
				use_simple = False
				X = age_imputer.fit_transform(X)
			except NameError as e:
				print(f"IterativeImputer failed due to missing import, imputing Age column with SimpleImputer instead.")
		if use_simple:
			age_imputer = SimpleImputer().set_output(transform="pandas")
			X["Age"] = age_imputer.fit_transform(X[["Age"]])

	X.columns = [c.replace("impute_nums__", "").replace("remainder__", "") for c in X.columns]

	if verbose:
		print("Scaling 'Age' and 'Fare' features...")
	if scaler_:
		scaler = scaler_
	else:
		scaler = ColumnTransformer(
			transformers=[
				('scaler', StandardScaler(), ["Age", "Fare"])
			],
			remainder='passthrough'
		).set_output(transform="pandas")

	X = scaler.fit_transform(X)
	X.columns = [c.replace("scaler__", "").replace("remainder__", "") for c in X.columns]

	if len(X[X["Fare"].isna()]) != 0:
		X["Fare"] = X["Fare"].fillna(X["Fare"].mean())

	if return_transformers:
		return X, y, embarked_encoder, pclass_encoder, age_imputer, scaler, use_simple
	else:
		return X, y


if __name__ == "__main__":
	cwd = os.getcwd() + "/"

	# if args.download_data:
	# 	pass
	# else:
	# 	raw_train_df = pd.read_csv(f"{cwd}data/train.csv")
	# 	test_df = pd.read_csv(f"{cwd}data/test.csv")
	raw_train_df = pd.read_csv(f"{cwd}data/train.csv")
	test_df = pd.read_csv(f"{cwd}data/test.csv")

	X, y, embarked_encoder, pclass_encoder, age_imputer, scaler, use_simple = prep_data(raw_train_df, return_transformers=True, verbose=True)
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=93)

	warnings.filterwarnings("ignore", category=ConvergenceWarning) #get rid of annoying convergence warning
	
	print("\nTraining model...")
	model = LogisticRegression(max_iter=250)
	model.fit(X_train,y_train)
	model.score(X_test,y_test)

	train_preds = model.predict(X_train)
	train_accuracy = accuracy_score(y_train, train_preds)
	train_f1 = f1_score(y_train, train_preds)
	train_precision = precision_score(y_train, train_preds)
	train_recall = recall_score(y_train, train_preds)

	print(f"\n---TRAIN-TEST SCORES---")
	print(f"Model training accuracy score: {round(train_accuracy, 4)}")
	print(f"Model training f1 score: {round(train_f1, 4)}")
	print(f"Model training precision score: {round(train_precision, 4)}")
	print(f"Model training recall score: {round(train_recall, 4)}")

	test_preds = model.predict(X_test)
	test_accuracy = accuracy_score(y_test, test_preds)
	test_f1 = f1_score(y_test, test_preds)
	test_precision = precision_score(y_test, test_preds)
	test_recall = recall_score(y_test, test_preds)

	print(f"\nModel test accuracy score: {round(test_accuracy, 4)}")
	print(f"Model test f1 score: {round(test_f1, 4)}")
	print(f"Model test precision score: {round(test_precision, 4)}")
	print(f"Model test recall score: {round(test_recall, 4)}")

	test_data, _ = prep_data(test_df, embarked_encoder, pclass_encoder, age_imputer, scaler, use_simple)
	final_test_preds = np.hstack([test_df["PassengerId"].to_numpy().reshape(-1,1), model.predict(test_data).reshape(-1,1)])
	
	if not args.hide_test_preds:
		print(f"\n---TEST PREDICTIONS---")
		print(final_test_preds)
		print("(Pass '-ht' / '--hide_test_preds' command line argument to only show train-test scores)")
	else:
		print("\nFinal test predictions hidden.")


