# -*- coding: utf-8 -*-
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef
from xgboost import XGBClassifier

THRESHOLD = 0.5


def evalmcc(preds, dtrain):
    labels = dtrain.get_label()
    return 'MCC', matthews_corrcoef(labels, preds > THRESHOLD)


def evalmcc_min(preds, dtrain):
    labels = dtrain.get_label()
    return 'MCC', -matthews_corrcoef(labels, preds > THRESHOLD)


xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.5,
    'silent': 1,
    'subsample': 0.5,
    'learning_rate': 0.001,
    'objective': 'binary:logistic',
    'max_depth': 2,
    'min_child_weight': 1,
}

if __name__ == "__main__":
    digits = load_digits(n_class=2)
    x_train = digits.data
    y_train = digits.target

    dtrain = xgb.DMatrix(x_train, label=y_train)
    res = xgb.cv(xgb_params, dtrain, num_boost_round=250, nfold=5, seed=0, stratified=True,
                 early_stopping_rounds=25, verbose_eval=5, show_stdv=True, feval=evalmcc, maximize=True)

    clf = XGBClassifier(**xgb_params)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.50, random_state=1337)

    clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric=evalmcc_min, early_stopping_rounds=50,
            verbose=True)
