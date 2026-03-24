import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from evaluation_metrics import get_classification_report, get_f1_score, get_auc_score, get_eer_score


# Default hyperparameter grid for SVM
DEFAULT_PARAM_GRID = {
    "svm__C":      [0.01, 0.1, 1, 10, 100],
    "svm__kernel": ["linear", "rbf"],
    "svm__gamma":  ["scale", "auto"],
}


def run_grid_search(
    X_train,
    X_test,
    y_train,
    y_test,
    param_grid=None,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1,
):
    """
    Run a GridSearchCV over an SVM classifier on pre-extracted CNN features.

    Parameters
    ----------
    X_train, X_test : np.ndarray  – feature matrices from feature extractor
    y_train, y_test : np.ndarray  – binary labels (0 = real, 1 = fake)
    param_grid      : dict        – sklearn param grid; defaults to DEFAULT_PARAM_GRID
    cv              : int         – number of cross-validation folds
    scoring         : str         – metric used to select the best model
    n_jobs          : int         – parallel jobs (-1 = all cores)
    verbose         : int         – verbosity level for GridSearchCV

    Returns
    -------
    dict with keys:
        best_estimator  – fitted Pipeline (scaler + SVM) with best params
        best_params     – dict of best hyperparameters
        best_cv_score   – best cross-validated score on the training set
        classification_report – sklearn classification report string
        f1              – F1 score on test set
        auc             – ROC-AUC on test set
        eer             – Equal Error Rate on test set
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    # Pipeline: scale features, then SVM
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(probability=True, random_state=42)),
    ])

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    final_preds = best_model.predict(X_test)
    y_probs     = best_model.predict_proba(X_test)[:, 1]

    return {
        "best_estimator":        best_model,
        "best_params":           grid_search.best_params_,
        "best_cv_score":         grid_search.best_score_,
        "classification_report": get_classification_report(y_test, final_preds),
        "f1":                    get_f1_score(y_test, final_preds),
        "auc":                   get_auc_score(y_test, y_probs),
        "eer":                   get_eer_score(y_test, y_probs),
    }


def print_results(results, model_name="Model"):
    """Pretty-print the results dict returned by run_grid_search."""
    print(f"\n{'='*60}")
    print(f"  GridSearch Results – {model_name}")
    print(f"{'='*60}")
    print(f"  Best params     : {results['best_params']}")
    print(f"  Best CV score   : {results['best_cv_score']:.4f}  (scoring=f1)")
    print(f"  Test F1         : {results['f1']:.4f}")
    print(f"  Test AUC        : {results['auc']:.4f}")
    print(f"  Test EER        : {results['eer']:.4f}")
    print(f"\n  Classification Report:\n{results['classification_report']}")
