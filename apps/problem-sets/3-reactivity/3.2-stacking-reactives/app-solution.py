from shiny.express import render, ui, input
from shiny import reactive
from data_import import df
from plots import plot_auc_curve, plot_precision_recall_curve, plot_score_distribution
from shinywidgets import render_plotly


@reactive.calc
def account_data():
    return df[df["account"] == input.account()]


# You can call reactive calculations from other reactive calculations.
# This is very useful for performance because the calculation will be
# minimally rerun.
@reactive.calc()
def character_filter():
    return account_data()[account_data()["text"].str.len().between(*input.chars())]


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
                return plot_auc_curve(
                    character_filter(), "is_electronics", "training_score"
                )
            else:
                return plot_precision_recall_curve(
                    character_filter(), "is_electronics", "training_score"
                )

        ui.input_select("metric", "Metric", choices=["ROC Curve", "Precision Recall"])

    with ui.card():
        ui.card_header("Model Scores")

        @render_plotly
        def score_dist():
            return plot_score_distribution(character_filter())
