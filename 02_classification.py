import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, classification_report,
                             ConfusionMatrixDisplay, RocCurveDisplay)


df = pd.read_csv("dataset/dataset_euskadi_clean.csv")
raw = pd.read_csv("dataset/dataset_euskadi_raw.csv", usecols=["host_id"])

assert len(df) == len(raw)
df["host_id"] = raw["host_id"].values

y = (df["host_is_superhost"] == "t").astype(int)
print(y.value_counts(normalize=True).round(3))

num_cols = [
    "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication",
    "review_scores_location", "review_scores_value",
    "number_of_reviews", "number_of_reviews_ltm", "reviews_per_month",
    "hosts_time_as_host_years", "calculated_host_listings_count",
    "accommodates", "bedrooms", "beds", "bathrooms_num",
    "num_amenities", "price_num", "minimum_nights", "availability_365",
]
cat_cols = ["neighbourhood_group_cleansed", "room_type",
            "host_identity_verified", "host_has_profile_pic"]

X = df[num_cols + cat_cols]
groups = df["host_id"]


splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
g_train, g_test = groups.iloc[train_idx], groups.iloc[test_idx]

print("train size:", len(X_train), "test size:", len(X_test))
print("hosts in both sets:", len(set(g_train) & set(g_test)))
print("superhost rate train:", round(y_train.mean(), 3), "test:", round(y_test.mean(), 3))


num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler()),
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])


cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]

baseline = Pipeline([("prep", preprocessor),
                     ("model", DummyClassifier(strategy="most_frequent"))])
logreg = Pipeline([("prep", preprocessor),
                   ("model", LogisticRegression(max_iter=1000))])

for name, pipe in [("Baseline", baseline), ("Logistic Regression (default)", logreg)]:
    scores = cross_validate(pipe, X_train, y_train, groups=g_train,
                            cv=cv, scoring=scoring)
    print(name)
    for metric in scoring:
        values = scores["test_" + metric]
        print("  ", metric, round(values.mean(), 3), "+-", round(values.std(), 3))


models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000),
        {"model__C": [0.01, 0.1, 1, 10, 100],
         "model__class_weight": [None, "balanced"]}),
    "KNN": (
        KNeighborsClassifier(),
        {"model__n_neighbors": [5, 11, 21, 41],
         "model__weights": ["uniform", "distance"]}),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {"model__max_depth": [3, 5, 8, None],
         "model__min_samples_leaf": [1, 5, 20]}),
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {"model__n_estimators": [200, 400],
         "model__max_depth": [5, 10, None],
         "model__min_samples_leaf": [1, 5]}),
}

if os.path.exists("gridsearch_results.joblib"):
    results = joblib.load("gridsearch_results.joblib")
else:
    results = {}
    for name, (model, grid) in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        gs = GridSearchCV(pipe, grid, cv=cv, scoring=scoring,
                          refit="f1", n_jobs=-1)
        gs.fit(X_train, y_train, groups=g_train)
        results[name] = gs
        print(name, "best F1:", round(gs.best_score_, 3), gs.best_params_)
    joblib.dump(results, "gridsearch_results.joblib")


rows = []
for name, gs in results.items():
    i = gs.best_index_
    row = {"model": name}
    for metric in scoring:
        row[metric] = round(gs.cv_results_["mean_test_" + metric][i], 3)
    row["f1_std"] = round(gs.cv_results_["std_test_f1"][i], 3)
    row["best_params"] = gs.best_params_
    rows.append(row)

cv_table = pd.DataFrame(rows).sort_values("f1", ascending=False)
print(cv_table.to_string(index=False))
cv_table.to_csv("classification_cv_results.csv", index=False)


fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, metric in zip(axes, ["f1", "roc_auc"]):
    data = []
    for gs in results.values():
        i = gs.best_index_
        data.append([gs.cv_results_["split%d_test_%s" % (k, metric)][i] for k in range(5)])
    ax.boxplot(data)
    ax.set_xticks(range(1, len(results) + 1))
    ax.set_xticklabels(list(results.keys()), rotation=20)
    ax.set_title(metric + " per fold")
plt.tight_layout()
plt.savefig("fig_cv_folds.png", dpi=200, bbox_inches="tight")
plt.show()


baseline.fit(X_train, y_train)
final_models = {"Baseline": baseline}
for name, gs in results.items():
    final_models[name] = gs.best_estimator_

rows = []
for name, model in final_models.items():
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    rows.append({
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, prob),
    })

test_table = pd.DataFrame(rows).round(3).sort_values("f1", ascending=False)
print(test_table.to_string(index=False))
test_table.to_csv("classification_test_results.csv", index=False)

best_model = results["Random Forest"].best_estimator_
print(classification_report(y_test, best_model.predict(X_test),
                            target_names=["Not Superhost", "Superhost"]))


fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test,
                                      display_labels=["Not Superhost", "Superhost"],
                                      cmap="Blues", ax=axes[0])
axes[0].set_title("Confusion matrix (Random Forest, test)")

for name, model in final_models.items():
    if name != "Baseline":
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=axes[1])
axes[1].plot([0, 1], [0, 1], "k--", label="Chance")
axes[1].set_title("ROC curves (test)")
axes[1].legend()
plt.tight_layout()
plt.savefig("fig_test_cm_roc.png", dpi=200, bbox_inches="tight")
plt.show()


pi = permutation_importance(best_model, X_test, y_test, scoring="f1",
                            n_repeats=20, random_state=42, n_jobs=-1)
importance = pd.Series(pi.importances_mean, index=X_test.columns).sort_values()

importance.tail(12).plot.barh(figsize=(7, 5))
plt.xlabel("Drop in F1 when the feature is shuffled")
plt.title("Permutation importance (Random Forest, test)")
plt.tight_layout()
plt.savefig("fig_importance.png", dpi=200, bbox_inches="tight")
plt.show()


cv_repeat = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=7)
rows = []
for name, gs in results.items():
    scores = cross_validate(clone(gs.best_estimator_), X, y, groups=groups,
                            cv=cv_repeat, scoring=scoring)
    row = {"model": name}
    for metric in scoring:
        values = scores["test_" + metric]
        row[metric] = "%.3f +- %.3f" % (values.mean(), values.std())
    rows.append(row)

repeat_table = pd.DataFrame(rows)
print(repeat_table.to_string(index=False))
repeat_table.to_csv("classification_repeated_cv_results.csv", index=False)
