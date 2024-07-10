from shiny.express import render, ui, input
from data_import import df
from plots import plot_auc_curve, plot_precision_recall_curve
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

# Cards are great for grouping related elements together.
with ui.card():
    ui.card_header("Model Metrics")

    @render_plotly
    def metric_plot():
        account_subset = df[df["account"] == input.account()]
        if input.metric() == "ROC Curve":
            return plot_auc_curve(account_subset, "is_electronics", "training_score")
        else:
            return plot_precision_recall_curve(
                account_subset, "is_electronics", "training_score"
            )

    ui.input_select("metric", "Metric", choices=["ROC Curve", "Precision Recall"])
