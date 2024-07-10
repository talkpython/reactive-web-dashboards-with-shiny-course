from shiny.express import render, ui, input
from data_import import df
from shiny import reactive
from plots import plot_auc_curve, plot_precision_recall_curve, plot_score_distribution
from shinywidgets import render_plotly


@reactive.calc
def account_data():
    return df[df["account"] == input.account()]


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

    ui.input_slider(
        "chars",
        "Text length",
        min=0,
        max=df["text"].str.len().max(),
        value=[500, 6000],
    )

with ui.layout_columns():

    with ui.card():
        ui.card_header("Model Metrics")

        @render_plotly
        def metric_plot():
            if input.metric() == "ROC Curve":
                return plot_auc_curve(account_data, "is_electronics", "training_score")
            else:
                return plot_precision_recall_curve(
                    account_data(), "is_electronics", "training_score"
                )

        ui.input_select("metric", "Metric", choices=["ROC Curve", "Precision Recall"])

    with ui.card():
        ui.card_header("Model Scores")

        @render_plotly
        def score_dist():
            return plot_score_distribution(account_data())
