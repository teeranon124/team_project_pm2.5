from dash import Dash, html, dcc, callback, Output, Input, State, callback_context
import plotly.express as px
import pandas as pd
from pycaret.time_series import *
import dash_bootstrap_components as dbc
import dash_leaflet as dl
from math import sqrt

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


# Load features and prepare data
def load_and_prepare_features():
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
            "r202_test_wifi": pd.read_csv(
                "7D_X_feature/export-r202_test_wifi-1h-6H.csv"
            ),
        },
    }

    # Convert timestamp and set as index
    for time_frame, data_dict in X_features.items():
        for loc, data in data_dict.items():
            data["timestamp"] = pd.to_datetime(data["timestamp"], format="mixed")
            data.set_index("timestamp", inplace=True)
            data.index = data.index.to_period("6H" if time_frame == "6H" else "D")

    return X_features


X_features = load_and_prepare_features()

# Historical data
historical_data_6H = {
    "jsps001": pd.read_csv("6H_train/export-jsps001-6H.csv"),
    "jsps013": pd.read_csv("6H_train/export-jsps013-1h-6H.csv"),
    "jsps014": pd.read_csv("6H_train/export-jsps014-1h-6H.csv"),
    "jsps018": pd.read_csv("6H_train/export-jsps018-1h.csv"),
    "pm25_eng": pd.read_csv("6H_train/export-pm25_eng-1h.csv"),
    "r202_test_wifi": pd.read_csv("6H_train/export-r202_test_wifi-1h.csv"),
}

historical_data_1D = {
    "jsps001": pd.read_csv("7D_train/export-jsps001-6H.csv"),
    "jsps013": pd.read_csv("7D_train/export-jsps013-1h-6H.csv"),
    "jsps014": pd.read_csv("7D_train/export-jsps014-1h-6H.csv"),
    "jsps018": pd.read_csv("7D_train/export-jsps018-1h-6H.csv"),
    "pm25_eng": pd.read_csv("7D_train/export-pm25_eng-1h-6H.csv"),
    "r202_test_wifi": pd.read_csv("7D_train/export-r202_test_wifi-1h-6H.csv"),
}

# Convert timestamps
for loc, data in historical_data_6H.items():
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="mixed")
    data.set_index("timestamp", inplace=True)
    data.index = data.index.to_period("6H")

for loc, data in historical_data_1D.items():
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="mixed")
    data.set_index("timestamp", inplace=True)
    data.index = data.index.to_period("D")

# Locations
locations = {
    "jsps001": {"name": "JSPS001", "lat": 9.134142, "lon": 99.334923},
    "jsps013": {"name": "JSPS013", "lat": 7.022061, "lon": 100.471286},
    "jsps014": {"name": "JSPS014", "lat": 7.020224, "lon": 100.470991},
    "jsps018": {"name": "JSPS018", "lat": 7.007673, "lon": 100.471091},
    "pm25_eng": {"name": "PM2.5 ENG", "lat": 7.007346, "lon": 100.501359},
    "r202_test_wifi": {"name": "R202 Test Wifi", "lat": 7.007346, "lon": 100.502197},
}

# Create Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# Air quality color function
def get_pm25_color(pm25):
    if pm25 <= 50:
        return "#4ade80"  # Green
    elif pm25 <= 100:
        return "#fbbf24"  # Yellow
    else:
        return "#ef4444"  # Red


import dash_leaflet as dl
import dash_html_components as html

markers = [
    dl.Marker(
        position=[locations[loc]["lat"], locations[loc]["lon"]],
        id=f"marker-{loc}",
        children=[
            dl.Tooltip(
                [
                    html.H4(f"{locations[loc]['name']}"),
                    html.P(
                        f"PM2.5: {round(historical_data_6H[loc].iloc[-1]['pm_2_5'], 2)}"
                    ),
                ],
                className="leaflet-popup-content",  # Apply the existing popup content class from your CSS
            ),
        ],
    )
    for loc in locations
]


# Marquee content
marquee_content = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.Span(f"{locations[loc]['name']}: "),
                        html.Span(
                            f"{round(historical_data_6H[loc].iloc[-1]['pm_2_5'], 2)}",
                            style={
                                "color": get_pm25_color(
                                    historical_data_6H[loc].iloc[-1]["pm_2_5"]
                                )
                            },
                        ),
                    ],
                    className="marquee-item",
                )
                for loc in locations
            ],
            className="marquee-content",
        )
    ],
    className="marquee",
)

# App layout
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1(
                    children="PM2.5 Prediction Dashboard", className="text-center my-4"
                )
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dl.Map(
                            [
                                dl.TileLayer(),
                                *markers,
                                dl.LayerGroup(id="map-click-layer"),
                            ],
                            id="map",
                            style={"height": "500px", "width": "100%"},
                            center=[7.007346, 100.502197],
                            zoom=12,
                        ),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(html.Label("Select Time Frame"), width=6),
                                dbc.Col(
                                    dcc.Dropdown(
                                        options=[
                                            {
                                                "label": "7 days, every 6 hours",
                                                "value": "6H",
                                            },
                                            {
                                                "label": "7 days, every 24 hours",
                                                "value": "1D",
                                            },
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
                                dbc.Col(html.Label("Select Locations"), width=6),
                                dbc.Col(
                                    dcc.Dropdown(
                                        options=[
                                            {
                                                "label": locations[loc]["name"],
                                                "value": loc,
                                            }
                                            for loc in locations
                                        ],
                                        value=["jsps001", "jsps013"],
                                        id="locations",
                                        multi=True,
                                        className="form-control mb-3",
                                    ),
                                    width=6,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(html.Label("Select Chart Type"), width=6),
                                dbc.Col(
                                    dcc.Dropdown(
                                        options=[
                                            {"label": "Line Chart", "value": "line"},
                                            {"label": "Bar Chart", "value": "bar"},
                                        ],
                                        value="line",
                                        id="chart_type",
                                        className="form-control mb-3",
                                    ),
                                    width=6,
                                ),
                            ]
                        ),
                        dbc.Row(
                            dbc.Col(
                                dbc.Button(
                                    "Predict",
                                    id="predict-button",
                                    color="primary",
                                    className="w-100",
                                ),
                                width=12,
                            )
                        ),
                    ],
                    width=6,
                ),
            ]
        ),
        dbc.Row(dbc.Col(marquee_content, width=12)),
        dbc.Modal(
            [
                dbc.ModalHeader("PM2.5 Prediction Graph"),
                dbc.ModalBody(dcc.Graph(id="prediction-graph")),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-button", className="ms-auto")
                ),
            ],
            id="prediction-modal",
            size="xl",
            is_open=False,
        ),
    ],
    fluid=True,
)


# Calculate distance between two points
def calculate_distance(lat1, lon1, lat2, lon2):
    return sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)


# Callback for map click events
@app.callback(
    Output("map-click-layer", "children"),  # Output: Popup content
    Input("map", "click_lat_lng"),  # Input: Map click coordinates
    State("time_frame", "value"),  # State: Selected time frame
)
def map_click(click_lat_lng, time_frame):
    if not click_lat_lng:
        return None

    click_lat, click_lon = click_lat_lng
    closest_location = None
    min_distance = float("inf")

    # Find the closest location to the clicked coordinates
    for loc, info in locations.items():
        distance = calculate_distance(click_lat, click_lon, info["lat"], info["lon"])
        if distance < min_distance:
            min_distance = distance
            closest_location = loc

    if closest_location:
        historical_data = (
            historical_data_6H if time_frame == "6H" else historical_data_1D
        )
        latest_data = historical_data[closest_location].iloc[-1]
        pm25 = latest_data["pm_2_5"]

        # Create popup content
        popup_content = html.Div(
            [
                html.H4(
                    f"Location: {locations[closest_location]['name']}"
                ),  # Location name
                html.P(f"PM2.5: {round(pm25, 2)}"),  # PM2.5 value
            ]
        )

        return dl.Popup(children=popup_content, position=[click_lat, click_lon])

    return None


# Callback for prediction
@app.callback(
    Output("prediction-modal", "is_open"),  # Output: Modal visibility
    Output("prediction-graph", "figure"),  # Output: Prediction graph
    Input("predict-button", "n_clicks"),  # Input: Predict button clicks
    Input("close-button", "n_clicks"),  # Input: Close button clicks
    State("time_frame", "value"),  # State: Selected time frame
    State("locations", "value"),  # State: Selected locations
    State("chart_type", "value"),  # State: Selected chart type
    State("prediction-modal", "is_open"),  # State: Modal visibility
)
def predict(
    n_clicks_predict,
    n_clicks_close,
    time_frame,
    selected_locations,
    chart_type,
    is_open,
):
    if n_clicks_predict is None and n_clicks_close is None:
        return False, {}

    ctx = callback_context
    if not ctx.triggered:
        return False, {}
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "predict-button":
        if not selected_locations:
            return False, {}

        predictions = []
        for loc in selected_locations:
            model = models[time_frame][loc]
            X_feature = X_features[time_frame][loc]
            try:
                prediction = predict_model(model, X=X_feature)
                predictions.append(
                    {
                        "location": loc,
                        "data": prediction["y_pred"],
                        "timestamps": X_feature.index.to_timestamp(),
                    }
                )
            except Exception as e:
                print(f"Error predicting for {loc}: {e}")
                continue

        if chart_type == "line":
            fig = px.line()
            for pred in predictions:
                historical = (
                    historical_data_6H[pred["location"]]
                    if time_frame == "6H"
                    else historical_data_1D[pred["location"]]
                )
                fig.add_scatter(
                    x=historical.index.to_timestamp(),
                    y=historical["pm_2_5"],
                    name=f"{locations[pred['location']]['name']} (Historical)",
                    mode="lines",
                    line=dict(dash="dash"),
                )
                fig.add_scatter(
                    x=pred["timestamps"],
                    y=pred["data"],
                    name=f"{locations[pred['location']]['name']} (Predicted)",
                    mode="lines",
                )
        else:
            fig = px.bar()
            for pred in predictions:
                historical = (
                    historical_data_6H[pred["location"]]
                    if time_frame == "6H"
                    else historical_data_1D[pred["location"]]
                )
                fig.add_bar(
                    x=historical.index.to_timestamp(),
                    y=historical["pm_2_5"],
                    name=f"{locations[pred['location']]['name']} (Historical)",
                )
                fig.add_bar(
                    x=pred["timestamps"],
                    y=pred["data"],
                    name=f"{locations[pred['location']]['name']} (Predicted)",
                )

        fig.update_layout(
            title="PM2.5 Prediction Comparison (7 Days Historical vs 7 Days Predicted)",
            xaxis_title="Time",
            yaxis_title="PM2.5 Prediction",
            legend_title="Locations",
        )

        return True, fig

    elif button_id == "close-button":
        return False, {}

    return False, {}


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
