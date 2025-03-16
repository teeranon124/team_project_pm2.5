from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
from pycaret.time_series import *
import dash_bootstrap_components as dbc

# Load models
models = {
    "6H": {
        "jsps001": load_model("models/6H_export-jsps001-1h"),
        "jsps013": load_model("models/6H_export-jsps013-1h"),
        "jsps014": load_model("models/6H_export-jsps014-1h"),
        "jsps018": load_model("models/6H_export-jsps018-1h"),
        "pm25_eng": load_model("models/6H_export-pm25_eng-1h"),
        "r202_test_wifi": load_model("models/6H_export-r202_test_wifi-1h"),
    },
    "1D": {
        "jsps001": load_model("models/export-jsps001-1h"),
        "jsps013": load_model("models/export-jsps013-1h"),
        "jsps014": load_model("models/export-jsps014-1h"),
        "jsps018": load_model("models/export-jsps018-1h"),
        "pm25_eng": load_model("models/export-pm25_eng-1h"),
        "r202_test_wifi": load_model("models/export-r202_test_wifi-1h"),
    },
}

# Load features
X_features = {
    "6H": {
        "jsps001": pd.read_csv("6H_X_feature/export-jsps001-6H.csv"),
        "jsps013": pd.read_csv("6H_X_feature/export-jsps013-1h-6H.csv"),
        "jsps014": pd.read_csv("6H_X_feature/export-jsps014-1h-6H.csv"),
        "jsps018": pd.read_csv("6H_X_feature/export-jsps018-1h.csv"),
        "pm25_eng": pd.read_csv("6H_X_feature/export-pm25_eng-1h.csv"),
        "r202_test_wifi": pd.read_csv("6H_X_feature/export-r202_test_wifi-1h.csv"),
    },
    "1D": {
        "jsps001": pd.read_csv("7D_X_feature/export-jsps001-6H.csv"),
        "jsps013": pd.read_csv("7D_X_feature/export-jsps013-1h-6H.csv"),
        "jsps014": pd.read_csv("7D_X_feature/export-jsps014-1h-6H.csv"),
        "jsps018": pd.read_csv("7D_X_feature/export-jsps018-1h-6H.csv"),
        "pm25_eng": pd.read_csv("7D_X_feature/export-pm25_eng-1h-6H.csv"),
        "r202_test_wifi": pd.read_csv("7D_X_feature/export-r202_test_wifi-1h-6H.csv"),
    },
}

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1(
                    children="PM2.5 Prediction Dashboard", className="text-center my-4"
                )
            ),
        ),
        dbc.Row(
            [
                dbc.Col(html.Label("Select Time Frame"), width=6),
                dbc.Col(
                    dcc.Dropdown(
                        options=[
                            {"label": "6 Hours", "value": "6H"},
                            {"label": "7 Days", "value": "1D"},
                        ],
                        value="6H",
                        id="time_frame",
                        className="form-control mb-3",
                    ),
                    width=6,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(html.Label("Select Location"), width=6),
                dbc.Col(
                    dcc.Dropdown(
                        options=[
                            {"label": "JSPS001", "value": "jsps001"},
                            {"label": "JSPS013", "value": "jsps013"},
                            {"label": "JSPS014", "value": "jsps014"},
                            {"label": "JSPS018", "value": "jsps018"},
                            {"label": "PM2.5 ENG", "value": "pm25_eng"},
                            {"label": "R202 Test Wifi", "value": "r202_test_wifi"},
                        ],
                        value="jsps001",
                        id="location",
                        className="form-control mb-3",
                    ),
                    width=6,
                ),
            ]
        ),
        dbc.Row(dbc.Col(html.Div(id="prediction-content", className="mt-4"))),
        dbc.Row(dbc.Col(dcc.Graph(id="prediction-graph", className="mt-4"))),
    ],
    fluid=True,
)


@callback(
    Output("prediction-content", "children"),
    Output("prediction-graph", "figure"),
    Input("time_frame", "value"),
    Input("location", "value"),
)
def predict(time_frame, location):
    if None in [time_frame, location]:
        return "Please provide all input values.", {}

    model = models[time_frame][location]
    X_feature = X_features[time_frame][location]

    prediction = predict_model(model, X=X_feature)
    prediction_label = prediction["prediction_label"]

    fig = px.line(
        x=X_feature.index,
        y=prediction_label,
        labels={"x": "Time", "y": "PM2.5 Prediction"},
        title=f"PM2.5 Prediction for {location} ({time_frame})",
    )

    return f"Prediction: {prediction_label[0]}", fig


if __name__ == "__main__":
    app.run_server(debug=True)
