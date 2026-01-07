"""
File: boston_housing_competition.py
Name: Kang
--------------------------------
This file demonstrates how to analyze boston
housing dataset. Students will upload their 
results to kaggle.com and compete with people
in class!

You are allowed to use pandas, sklearn, or build the
model from scratch! Go data scientists!
"""

import pandas as pd
from sklearn import preprocessing, linear_model, metrics, model_selection, svm
from xgboost import XGBRegressor

TRAIN_PATH = 'boston_housing/train.csv'
TEST_PATH = 'boston_housing/test.csv'


def main():
	train_data = pd.read_csv(TRAIN_PATH)
	test_data = pd.read_csv(TEST_PATH)

	voting_model(train_data, test_data)


def voting_model(train_data, test_data):
	y = train_data.pop('medv')

	x_train, x_valid, y_train, y_valid = model_selection.train_test_split(train_data, y, test_size=0.3, random_state=5)

	# ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'lstat']
	features = ['chas', 'nox', 'rm', 'rad', 'tax', 'ptratio', 'lstat']
	x_train = x_train[features]
	x_valid = x_valid[features]

	# ridge + robust
	scaler = preprocessing.RobustScaler()
	poly = preprocessing.PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
	x_train_scaler = scaler.fit_transform(x_train)
	x_train_poly = poly.fit_transform(x_train_scaler)

	ridge = linear_model.Ridge(alpha=0.03, solver='auto')
	ridge.fit(x_train_poly, y_train)
	predictor_ridge = ridge.predict(x_train_poly)
	# ridge_valid
	x_valid_scaler = scaler.transform(x_valid)
	x_valid_poly = poly.transform(x_valid_scaler)
	ridge_val_predictions = ridge.predict(x_valid_poly)

	# xgb model
	xgb = XGBRegressor(n_estimators=65, max_depth=3, learning_rate=0.2, random_state=5)
	xgb.fit(x_train, y_train)
	predictor_xgb = xgb.predict(x_train)
	xgb_val_predictions = xgb.predict(x_valid)

	# SVR
	svr = svm.SVR(C=60, epsilon=0.05, kernel='rbf', gamma=0.34, tol=0.8)
	svr.fit(x_train_scaler, y_train)
	predictor_svr = svr.predict(x_train_scaler)
	svr_val_predictions = svr.predict(x_valid_scaler)

	# voting
	weights = [(0.3, 0.4, 0.3), (0.2, 0.5, 0.3), (0.4, 0.4, 0.2), (0.45, 0.45, 0.1), (0.2, 0.6, 0.2),
			   (0.15, 0.6, 0.25), (0.33, 0.33, 0.34), (0.25, 0.25, 0.5)]
	best_rmse = float('inf')
	best_weight = (0.3, 0.4, 0.3)

	for w_xgb, w_ridge, w_svr in weights:
		blend_pred = (
				w_xgb * xgb_val_predictions +
				w_ridge * ridge_val_predictions +
				w_svr * svr_val_predictions
		)
		train_pred = (w_xgb * predictor_xgb + w_ridge * predictor_ridge + w_svr * predictor_svr)
		train_rmse = metrics.mean_squared_error(y_train, train_pred, squared=False)
		rmse = metrics.mean_squared_error(y_valid, blend_pred, squared=False)
		print(f"RMSE (XGB: {w_xgb}, Ridge: {w_ridge}, SVR: {w_svr}) = train: {train_rmse:.4f} | valid: {rmse:.4f}")

		if rmse < best_rmse:
			best_rmse = rmse
			best_weight = (w_xgb, w_ridge, w_svr)

	# --- 測試資料 ---
	x_test = test_data[features]
	y_id = test_data.pop('ID')

	x_test_scaled = scaler.transform(x_test)
	x_test_poly = poly.transform(x_test_scaled)
	ridge_test_pred = ridge.predict(x_test_poly)
	xgb_test_pred = xgb.predict(x_test)

	svr_test_pred = svr.predict(x_test_scaled)

	# 最佳權重組合預測
	final_test_pred = (
			best_weight[0] * xgb_test_pred +
			best_weight[1] * ridge_test_pred +
			best_weight[2] * svr_test_pred)
	print(f"best weight: XGB: {best_weight[0]}, Ridge: {best_weight[1]}, SVR: {best_weight[2]}")
	# 輸出檔案
	out_file(final_test_pred, 'voting_0802.csv', y_id)


def out_file(predictions, filename, y_id):
	"""
	: param predictions: numpy.array, a list-like data structure that stores 0's and 1's
	: param filename: str, the filename you would like to write the results to
	"""
	print('\n===============================================')
	print(f'Writing predictions to --> {filename}')
	with open(filename, 'w') as out:
		out.write('ID,medv\n')
		for i, ans in enumerate(predictions):
			out.write(str(y_id[i])+','+str(ans.round(8))+'\n')
	print('===============================================')


if __name__ == '__main__':
	main()
