from shutil import ignore_patterns
from shiny.express import render, ui, input
from shiny import reactive
from data_import import df
from plots import plot_auc_curve, plot_precision_recall_curve, plot_score_distribution
from shinywidgets import render_plotly


@reactive.calc()
def account_data():
    return df[
        (df["account"] == input.account()) & (df["sub_account"] == input.sub_account())
    ]


# You can control execution by adding `@reactive.event` to any reactive function.
# This tells Shiny to only trigger this renderer when a particular input changes.
# Note that when inputs are used in reactive.events they are not called (input.update not input.update())
# This is because we are watching the reactive itself for changes, not making use of the current value.
@reactive.calc()
@reactive.event(input.update, ignore_init=True)
def character_filter():
    return account_data()[(account_data()["text"].str.len().between(*input.chars()))]


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

    @render.express
    def sub_selector():
        # We can't use the account_data reactive here because we are using the
        # sub_account input as part of that reactive.
        choice_data = df[df["account"] == input.account()]
        choices = choice_data["sub_account"].unique().tolist()
        ui.input_select("sub_account", "Sub Account", choices=choices)

    ui.input_slider(
        "chars",
        "Text length",
        min=0,
        max=df["text"].str.len().max(),
        value=[500, 6000],
    )

    ui.input_action_button("update", "Update", class_="btn-primary")

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
