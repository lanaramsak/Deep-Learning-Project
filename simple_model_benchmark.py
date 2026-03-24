
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evaluation_metrics import get_auc_score, get_eer_score, get_f1_score

EXTRACTOR_CHOICES = ["resnet18", "resnet50", "mobilenet", "inceptionv3", "efficientnet"]


@dataclass
class ModelConfig:
    name: str
    estimator: object
    use_scaler: bool = False
    use_pca: bool = False


def get_optional_model_configs():
    configs = []

    try:
        from xgboost import XGBClassifier
        configs.append(
            ModelConfig(
                name="XGBoost",
                estimator=XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=42,
                ),
                use_scaler=False,
                use_pca=False,
            )
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier
        configs.append(
            ModelConfig(
                name="LightGBM",
                estimator=LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=42,
                    verbose=-1,
                ),
                use_scaler=False,
                use_pca=False,
            )
        )
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier
        configs.append(
            ModelConfig(
                name="CatBoost",
                estimator=CatBoostClassifier(
                    iterations=300,
                    learning_rate=0.05,
                    depth=6,
                    random_seed=42,
                    verbose=0,
                ),
                use_scaler=False,
                use_pca=False,
            )
        )
    except ImportError:
        pass

    return configs


def get_model_configs():
    configs = [
        ModelConfig(
            name="LinearRegression",
            estimator=LinearRegression(),
            use_scaler=True,
            use_pca=True,
        ),
        ModelConfig(
            name="LogisticRegression",
            estimator=LogisticRegression(max_iter=2000),
            use_scaler=True,
            use_pca=True,
        ),
        ModelConfig(
            name="KNN",
            estimator=KNeighborsClassifier(n_neighbors=5),
            use_scaler=True,
            use_pca=True,
        ),
        ModelConfig(
            name="GaussianNB",
            estimator=GaussianNB(),
            use_scaler=True,
            use_pca=True,
        ),
        ModelConfig(
            name="SVM-RBF",
            estimator=SVC(kernel="rbf"),
            use_scaler=True,
            use_pca=True,
        ),
        ModelConfig(
            name="RandomForest",
            estimator=RandomForestClassifier(n_estimators=300, random_state=42),
            use_scaler=False,
            use_pca=False,
        ),
        ModelConfig(
            name="MLP",
            estimator=MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
            use_scaler=True,
            use_pca=True,
        ),
    ]
    configs.extend(get_optional_model_configs())
    return configs


def build_pipeline(config, pca_components):
    steps = []
    if config.use_scaler:
        steps.append(("scaler", StandardScaler()))
    if config.use_pca:
        steps.append(("pca", PCA(n_components=pca_components)))
    steps.append(("model", config.estimator))
    return Pipeline(steps)


def get_dataset(extractor_name):
    if extractor_name == "resnet18":
        from ResNet18 import extract_subsets_ResNet18
        return extract_subsets_ResNet18()
    if extractor_name == "resnet50":
        from ResNet50 import extract_subsets_ResNet50
        return extract_subsets_ResNet50()
    if extractor_name == "mobilenet":
        from MobileNet import extract_subsets_MobileNet
        return extract_subsets_MobileNet()
    if extractor_name == "inceptionv3":
        from InceptionV3 import extract_subsets_InceptionV3
        return extract_subsets_InceptionV3()
    if extractor_name == "efficientnet":
        from EfficientNet import extract_subsets_EfficientNet
        return extract_subsets_EfficientNet()
    raise ValueError(f"Unknown extractor: {extractor_name}")


def normalize_pca_components(value):
    if value > 1 and value.is_integer():
        return int(value)
    return value


def get_score_vector(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)
    return model.predict(X_test)


def evaluate_model(config, X_train, X_test, y_train, y_test, pca_components):
    model = build_pipeline(config, pca_components)
    model.fit(X_train, y_train)

    raw_scores = get_score_vector(model, X_test)
    predictions = model.predict(X_test)

    if config.name == "LinearRegression":
        predictions = (raw_scores >= 0.5).astype(int)

    result = {
        "model": config.name,
        "f1": get_f1_score(y_test, predictions),
        "auc": get_auc_score(y_test, raw_scores),
        "eer": get_eer_score(y_test, raw_scores),
        "report": classification_report(y_test, predictions, digits=4),
    }

    pca_step = model.named_steps.get("pca")
    if pca_step is not None:
        result["pca_components"] = pca_step.n_components_
        result["explained_variance"] = float(pca_step.explained_variance_ratio_.sum())
    else:
        result["pca_components"] = None
        result["explained_variance"] = None

    return result


def format_summary(results):
    lines = []
    header = f"{'Model':<20} {'F1':>8} {'AUC':>8} {'EER':>8} {'PCA comps':>10} {'Expl.Var':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for result in sorted(results, key=lambda item: item["auc"], reverse=True):
        pca_components = "-" if result["pca_components"] is None else str(result["pca_components"])
        explained_variance = "-"
        if result["explained_variance"] is not None:
            explained_variance = f"{result['explained_variance']:.4f}"

        lines.append(
            f"{result['model']:<20} "
            f"{result['f1']:>8.4f} "
            f"{result['auc']:>8.4f} "
            f"{result['eer']:>8.4f} "
            f"{pca_components:>10} "
            f"{explained_variance:>10}"
        )

    return "\n".join(lines)


def save_results(output_dir, summaries):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"simple_models_summary_{timestamp}.txt"

    content = ["All Extractors Summary", ""]
    content.extend(summaries)
    output_path.write_text("\n".join(content))
    return output_path


def run_benchmark(extractor_name, pca_components, output_dir, save_report=True, print_top_model=True):
    X_train, X_test, y_train, y_test = get_dataset(extractor_name)
    configs = get_model_configs()

    results = []
    for config in configs:
        print(f"Running {extractor_name} | {config.name}...")
        result = evaluate_model(config, X_train, X_test, y_train, y_test, pca_components)
        results.append(result)

    print()
    print(f"Summary for {extractor_name}")
    print(format_summary(results))

    if print_top_model:
        print()
        best = max(results, key=lambda item: item["auc"])
        print(f"Top model for {extractor_name}")
        print(f"{best['model']} | F1={best['f1']:.4f} | AUC={best['auc']:.4f} | EER={best['eer']:.4f}")

    return results


def parse_args():
    parser = ArgumentParser(description="Benchmark simple ML models on extracted CNN features.")
    parser.add_argument(
        "--pca-components",
        type=float,
        default=0.95,
        help="PCA configuration for models that use PCA. Example: 0.95 or 150.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory for the saved report file.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summaries and skip saving detailed text reports.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pca_components = normalize_pca_components(args.pca_components)
    save_report = not args.summary_only
    summaries = []

    for idx, extractor_name in enumerate(EXTRACTOR_CHOICES):
        if idx > 0:
            print()
            print("=" * 72)
            print()
        results = run_benchmark(
            extractor_name=extractor_name,
            pca_components=pca_components,
            output_dir=args.output_dir,
            save_report=save_report,
            print_top_model=False,
        )
        summaries.append(f"Summary for {extractor_name}")
        summaries.append(format_summary(results))
        summaries.append("")

    if save_report:
        output_path = save_results(args.output_dir, summaries)
        print()
        print(f"Summary saved to: {output_path}")


if __name__ == "__main__":
    main()
