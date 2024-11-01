from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import autolgb

X, y = make_classification(
    n_samples=10_000, n_features=10, n_informative=5, n_classes=5
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)
dtrain = autolgb.Dataset(data=X_train, label=y_train)
dtest = autolgb.Dataset(data=X_test, label=y_test)

engine = autolgb.Engine()
engine.optimize(dtrain, ntrial=10)
engine.fit(dtrain)
engine.cv(dtrain)
engine.predict(dtest)
