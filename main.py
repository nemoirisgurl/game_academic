import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import joblib

ID_COL = "student_id"
GRADE_ORDER = ["A", "B+", "B", "C+", "C", "D+", "D", "F"]


def enable_responsive_layout(fig):
    def _on_resize(event):
        event.canvas.draw_idle()

    fig.canvas.mpl_connect("resize_event", _on_resize)
    return fig


def create_figure():
    fig = plt.figure(layout="constrained")
    return enable_responsive_layout(fig)


def create_subplots(nrows, ncols):
    fig, axes = plt.subplots(nrows, ncols, layout="constrained")
    enable_responsive_layout(fig)
    return fig, axes


def show_graph(title, x_label, y_label, rotate_x=False):
    plt.title(title, pad=10, fontsize=12, fontweight="bold")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if rotate_x:
        plt.xticks(rotation=20, ha="right")
    plt.show()


def classify_grade(score):
    if score >= 80:
        return "A"
    if score >= 75:
        return "B+"
    if score >= 70:
        return "B"
    if score >= 65:
        return "C+"
    if score >= 60:
        return "C"
    if score >= 55:
        return "D+"
    if score >= 50:
        return "D"
    return "F"


def ordered_grade_counts(series):
    counts = series.value_counts().reindex(GRADE_ORDER, fill_value=0)
    return counts.rename_axis("grade_class").reset_index(name="count")


def main():
    sns.set_theme(style="whitegrid", context="paper")
    data = pd.read_csv("Gaming_Academic_Performance.csv")

    corr = data.corr(numeric_only=True)
    create_figure()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        annot=True,
        cmap="RdYlBu_r",
        fmt=".2f",
        linewidths=0.5,
        mask=mask,
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
    )
    show_graph("Feature Correlation Heatmap", "", "")

    numeric_features = [
        "gaming_hours",
        "sleep_hours",
        "study_hours",
        "device_usage",
        "addiction_score",
        "reaction_time_ms",
        "attendance",
    ]
    categorical_features = ["stress_level"]
    features = numeric_features + categorical_features

    X = data.copy()
    y = data["grades"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train.to_csv("train.csv", index=False)
    X_test.to_csv("test.csv", index=False)

    student_id_train = X_train.pop(ID_COL)
    student_id_test = X_test.pop(ID_COL)

    X_train = X_train[features]
    X_test = X_test[features]

    y_train = y_train.clip(lower=0, upper=100)
    y_test = y_test.clip(lower=0, upper=100)
    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocess", transformer),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    max_depth=10,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    predictions = model.predict(X_test)
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.named_steps["regressor"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    create_figure()
    sns.scatterplot(x=y_test, y=predictions, alpha=0.6, color="#C0392B", edgecolor=None)
    plt.plot(
        [min_val, max_val], [min_val, max_val], color="#1F3A5F", lw=2, linestyle="--"
    )
    show_graph("Actual vs Predicted Grades", "Actual Grades", "Predicted Grades")

    train_mse = mean_squared_error(y_train, train_predictions)
    train_r2 = r2_score(y_train, train_predictions)
    train_rmse = np.sqrt(train_mse)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mse)
    train_residuals = y_train - train_predictions
    residuals = y_test - predictions
    print(f"Train Root Mean Squared Error: {train_rmse}")
    print(f"Train R^2 Score: {train_r2}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R^2 Score: {r2}")

    output = pd.DataFrame(
        {
            "student_id": student_id_test,
            "actual_grades": y_test,
            "predicted_grades": predictions,
        }
    )
    output["actual_grade_class"] = output["actual_grades"].apply(classify_grade)
    output["predicted_grade_class"] = output["predicted_grades"].apply(classify_grade)
    class_accuracy = accuracy_score(
        output["actual_grade_class"], output["predicted_grade_class"]
    )
    print(f"Letter Grade Accuracy: {class_accuracy:.4f}")

    class_distribution = pd.concat(
        [
            ordered_grade_counts(output["actual_grade_class"]).assign(dataset="Actual"),
            ordered_grade_counts(output["predicted_grade_class"]).assign(
                dataset="Predicted"
            ),
        ],
        ignore_index=True,
    )

    create_figure()
    sns.barplot(
        data=class_distribution,
        x="grade_class",
        y="count",
        hue="dataset",
        palette=["#2E86AB", "#F18F01"],
    )
    show_graph(
        "Actual vs Predicted Letter Grade Distribution",
        "Letter Grade",
        "Student Count",
    )

    confusion = pd.crosstab(
        pd.Categorical(
            output["actual_grade_class"], categories=GRADE_ORDER, ordered=True
        ),
        pd.Categorical(
            output["predicted_grade_class"], categories=GRADE_ORDER, ordered=True
        ),
        dropna=False,
    )
    create_figure()
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Student Count"},
    )
    show_graph("Letter Grade Confusion Matrix", "Predicted Class", "Actual Class")

    create_figure()
    sns.barplot(
        data=feature_importance,
        x="importance",
        y="feature",
        hue="feature",
        palette="crest",
        legend=False,
    )
    show_graph("Feature Importance for Grade Prediction", "Importance", "Feature")

    create_figure()
    sns.scatterplot(
        x=predictions, y=residuals, alpha=0.6, color="#6C5CE7", edgecolor=None
    )
    plt.axhline(0, color="#1F3A5F", linestyle="--", linewidth=2)
    show_graph("Residual Plot", "Predicted Grades", "Actual - Predicted")

    fig, axes = create_subplots(2, 2)

    metric_frame = pd.DataFrame(
        {
            "split": ["Train", "Test"],
            "R2": [train_r2, r2],
            "RMSE": [train_rmse, rmse],
        }
    )
    metric_long = metric_frame.melt(
        id_vars="split", var_name="metric", value_name="value"
    )
    sns.barplot(
        data=metric_long,
        x="metric",
        y="value",
        hue="split",
        palette=["#2E86AB", "#F18F01"],
        ax=axes[0, 0],
    )
    axes[0, 0].set_title(
        "Train vs Test Metrics", pad=12, fontsize=14, fontweight="bold"
    )
    axes[0, 0].set_xlabel("Metric")
    axes[0, 0].set_ylabel("Value")

    train_min = min(y_train.min(), train_predictions.min())
    train_max = max(y_train.max(), train_predictions.max())
    sns.scatterplot(
        x=y_train,
        y=train_predictions,
        alpha=0.35,
        color="#2E86AB",
        edgecolor=None,
        ax=axes[0, 1],
    )
    axes[0, 1].plot(
        [train_min, train_max],
        [train_min, train_max],
        color="#1F3A5F",
        linestyle="--",
        linewidth=2,
    )
    axes[0, 1].set_title(
        "Train Actual vs Predicted", pad=12, fontsize=14, fontweight="bold"
    )
    axes[0, 1].set_xlabel("Actual Grades")
    axes[0, 1].set_ylabel("Predicted Grades")

    sns.scatterplot(
        x=y_test,
        y=predictions,
        alpha=0.45,
        color="#F18F01",
        edgecolor=None,
        ax=axes[1, 0],
    )
    axes[1, 0].plot(
        [min_val, max_val],
        [min_val, max_val],
        color="#1F3A5F",
        linestyle="--",
        linewidth=2,
    )
    axes[1, 0].set_title(
        "Test Actual vs Predicted", pad=12, fontsize=14, fontweight="bold"
    )
    axes[1, 0].set_xlabel("Actual Grades")
    axes[1, 0].set_ylabel("Predicted Grades")

    sns.kdeplot(
        train_residuals,
        fill=True,
        color="#2E86AB",
        alpha=0.35,
        label="Train",
        ax=axes[1, 1],
    )
    sns.kdeplot(
        residuals,
        fill=True,
        color="#F18F01",
        alpha=0.35,
        label="Test",
        ax=axes[1, 1],
    )
    axes[1, 1].axvline(0, color="#1F3A5F", linestyle="--", linewidth=2)
    axes[1, 1].set_title(
        "Residual Distribution: Train vs Test", pad=12, fontsize=14, fontweight="bold"
    )
    axes[1, 1].set_xlabel("Actual - Predicted")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].legend()

    plt.show()

    train_sizes, train_scores, validation_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="r2",
        train_sizes=np.linspace(0.2, 1.0, 5),
    )
    learning_curve_frame = pd.DataFrame(
        {
            "train_size": np.concatenate([train_sizes, train_sizes]),
            "score": np.concatenate(
                [train_scores.mean(axis=1), validation_scores.mean(axis=1)]
            ),
            "split": ["Train"] * len(train_sizes) + ["Validation"] * len(train_sizes),
        }
    )

    create_figure()
    sns.lineplot(
        data=learning_curve_frame,
        x="train_size",
        y="score",
        hue="split",
        marker="o",
        palette=["#2E86AB", "#F18F01"],
    )
    show_graph("Learning Curve (R^2)", "Training Samples", "Score")

    output.to_csv("predicted_grades.csv", index=False)
    joblib.dump(model, "grade_prediction_model.pkl")
    joblib.dump(list(feature_names), "model_features.pkl")


if __name__ == "__main__":
    main()
