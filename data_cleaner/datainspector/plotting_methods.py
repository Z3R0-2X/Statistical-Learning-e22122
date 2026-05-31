import plotly.express as px


class PlottingMethods:

    def plot_bar_chart(
            self,
            df,
            column):

        counts = (
            df[column]
            .value_counts()
            .reset_index()
        )

        counts.columns = [
            column,
            "Count"
        ]

        fig = px.bar(

            counts,

            x=column,

            y="Count",

            title=f"Bar Chart - {column}"
        )

        fig.show()


    def plot_pie_chart(
            self,
            df,
            column):

        fig = px.pie(

            df,

            names=column,

            title=f"Pie Chart - {column}"
        )

        fig.show()


    def plot_histogram(
            self,
            df,
            column):

        fig = px.histogram(

            df,

            x=column,

            title=f"Histogram - {column}"
        )

        fig.show()