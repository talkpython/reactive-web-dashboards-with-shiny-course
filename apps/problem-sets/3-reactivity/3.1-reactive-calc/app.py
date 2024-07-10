from shiny.express import render, ui, input
from shiny import reactive
from data_import import df
from plots import plot_auc_curve, plot_precision_recall_curve, plot_score_distribution
from shinywidgets import render_plotly

with ui.sidebar():
    ui.input_select(
        "account",
        "Account",
        choices=[
            "Berge & Berge",
            "Fritsch & Fritsch",
            "Hintz & Hintz",
            "Mosciski and Sons",
            "Wolff Ltd",
        ],
    )

with ui.layout_columns():

    with ui.card():
        ui.card_header("Model Metrics")

        @render_plotly
        def metric_plot():
            account_subset = df[df["account"] == input.account()]
            if input.metric() == "ROC Curve":
                return plot_auc_curve(
                    account_subset, "is_electronics", "training_score"
                )
            else:
                return plot_precision_recall_curve(
                    account_subset, "is_electronics", "training_score"
                )

        ui.input_select("metric", "Metric", choices=["ROC Curve", "Precision Recall"])

    with ui.card():
        ui.card_header("Model Scores")

        @render_plotly
        def score_dist():
            account_subset = df[df["account"] == input.account()]
            return plot_score_distribution(account_subset)
