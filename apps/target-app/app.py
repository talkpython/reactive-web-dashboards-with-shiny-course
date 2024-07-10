from numpy import char, place
from shiny.express import ui, input, render
from shiny import reactive

from plots import (
    plot_score_distribution,
    plot_auc_curve,
    plot_precision_recall_curve,
)
import faicons as fa
import io
from shinywidgets import render_plotly
from data_import import df


@reactive.calc()
def account_data():
    return df[
        (df["account"] == input.account()) & (df["sub_account"] == input.sub_account())
    ]


@reactive.calc()
def character_filter():
    return account_data()[(account_data()["text"].str.len().between(*input.chars()))]


@reactive.effect
@reactive.event(input.reset)
def reset_vals():
    ui.update_slider("chars", value=[500, 6000])


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
        choice_data = df[df["account"] == input.account()]
        choices = choice_data["sub_account"].unique().tolist()
        ui.input_select("sub_account", "Sub Account", choices=choices)

    with ui.tooltip(id="btn_tooltip", placement="right"):
        ui.input_slider(
            "chars",
            "Text length",
            min=0,
            max=df["text"].str.len().max(),
            value=[500, 6000],
        )
        "The number of characters in the text"

    ui.input_action_button("reset", "Reset Values", class_="btn-primary")

with ui.nav_panel("Training Dashboard"):
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

            ui.input_select(
                "metric", "Metric", choices=["ROC Curve", "Precision Recall"]
            )

        with ui.card():
            ui.card_header("Model Scores")

            @render_plotly
            def score_dist():
                return plot_score_distribution(character_filter())


vb_theme = "bg-blue"

with ui.nav_panel("Data"):
    with ui.layout_columns():
        with ui.value_box(theme=vb_theme):
            "Number of records"

            @render.text
            def data_count():
                return str(character_filter().shape[0])

        with ui.value_box(theme=vb_theme):
            "Mean Score"

            @render.text
            def mean_score(theme=vb_theme):
                return f"{character_filter()['training_score'].mean() * 100:.2f}%"

        with ui.value_box(theme=vb_theme):
            "Mean Text Length"

            @render.text
            def mean_text_length():
                return f"{character_filter()['text'].str.len().mean():.2f} characters"

    with ui.card(full_screen=True):
        with ui.card_header():
            "Data"
            with ui.popover(
                title="Download",
                class_="d-inline-block pull-right",
            ):
                fa.icon_svg("download")

                @render.download(filename=lambda: f"{input.account()}_scores.csv")
                def download_data():
                    with io.BytesIO() as buf:
                        account_data().to_csv(buf, index=False)
                        buf.seek(0)
                        yield buf.getvalue()

        @render.data_frame
        def data_output():
            return account_data().drop(columns=["text"])
