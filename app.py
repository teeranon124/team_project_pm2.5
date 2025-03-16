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


# Load features และแปลง timestamp
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

    # แปลง timestamp และตั้งค่าเป็น index
    for time_frame, data_dict in X_features.items():
        for loc, data in data_dict.items():
            data["timestamp"] = pd.to_datetime(data["timestamp"], format="mixed")
            data.set_index("timestamp", inplace=True)
            data.index = data.index.to_period("6H" if time_frame == "6H" else "D")

    return X_features


X_features = load_and_prepare_features()

# ข้อมูล 6 ชั่วโมงย้อนหลัง (7 วัน)
historical_data_6H = {
    "jsps001": pd.read_csv("6H_train/export-jsps001-6H.csv"),
    "jsps013": pd.read_csv("6H_train/export-jsps013-1h-6H.csv"),
    "jsps014": pd.read_csv("6H_train/export-jsps014-1h-6H.csv"),
    "jsps018": pd.read_csv("6H_train/export-jsps018-1h.csv"),
    "pm25_eng": pd.read_csv("6H_train/export-pm25_eng-1h.csv"),
    "r202_test_wifi": pd.read_csv("6H_train/export-r202_test_wifi-1h.csv"),
}

# ข้อมูล 7 วันย้อนหลัง
historical_data_1D = {
    "jsps001": pd.read_csv("7D_train/export-jsps001-6H.csv"),
    "jsps013": pd.read_csv("7D_train/export-jsps013-1h-6H.csv"),
    "jsps014": pd.read_csv("7D_train/export-jsps014-1h-6H.csv"),
    "jsps018": pd.read_csv("7D_train/export-jsps018-1h-6H.csv"),
    "pm25_eng": pd.read_csv("7D_train/export-pm25_eng-1h-6H.csv"),
    "r202_test_wifi": pd.read_csv("7D_train/export-r202_test_wifi-1h-6H.csv"),
}

# แปลง timestamp ในข้อมูล 6 ชั่วโมงย้อนหลัง
for loc, data in historical_data_6H.items():
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="mixed")
    data.set_index("timestamp", inplace=True)
    data.index = data.index.to_period("6H")

# แปลง timestamp ในข้อมูล 7 วันย้อนหลัง
for loc, data in historical_data_1D.items():
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="mixed")
    data.set_index("timestamp", inplace=True)
    data.index = data.index.to_period("D")

# สถานที่และตำแหน่ง
locations = {
    "jsps001": {"name": "JSPS001", "lat": 13.7563, "lon": 100.5018},
    "jsps013": {"name": "JSPS013", "lat": 13.7463, "lon": 100.5118},
    "jsps014": {"name": "JSPS014", "lat": 13.7363, "lon": 100.5218},
    "jsps018": {"name": "JSPS018", "lat": 13.7263, "lon": 100.5318},
    "pm25_eng": {"name": "PM2.5 ENG", "lat": 13.7163, "lon": 100.5418},
    "r202_test_wifi": {"name": "R202 Test Wifi", "lat": 13.7063, "lon": 100.5518},
}

# สร้าง Dash App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# สร้าง Marker พร้อม Tooltip ที่แสดงชื่อและค่า pm_2_5 (ทศนิยม 2 ตำแหน่ง)
markers = [
    dl.Marker(
        position=[locations[loc]["lat"], locations[loc]["lon"]],
        children=dl.Tooltip(
            f"{locations[loc]['name']} - PM2.5: {round(historical_data_6H[loc].iloc[-1]['pm_2_5'], 2)}"
        ),
    )
    for loc in locations
]

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
                dbc.Col(
                    [
                        dl.Map(
                            [
                                dl.TileLayer(),
                                *markers,  # ใช้ markers ที่สร้างใหม่
                                dl.LayerGroup(id="map-click-layer"),
                            ],
                            id="map",
                            style={"height": "500px", "width": "100%"},
                            center=[13.7363, 100.5218],
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
        # เพิ่ม Modal สำหรับแสดงกราฟ
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


# ฟังก์ชันคำนวณระยะทางระหว่างสองจุด (lat, lon)
def calculate_distance(lat1, lon1, lat2, lon2):
    return sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)


# Callback สำหรับการคลิกบนแผนที่
@app.callback(
    Output("map-click-layer", "children"),  # อัปเดต LayerGroup เพื่อแสดง Popup
    Input("map", "click_lat_lng"),  # รับค่าตำแหน่งที่คลิกบนแผนที่
    State("time_frame", "value"),  # รับค่า time_frame จาก Dropdown
)
def map_click(click_lat_lng, time_frame):
    if not click_lat_lng:
        return None

    # ค่าตำแหน่งที่คลิก
    click_lat, click_lon = click_lat_lng

    # ค้นหาสถานที่ที่ใกล้ที่สุด
    closest_location = None
    min_distance = float("inf")

    for loc, info in locations.items():
        distance = calculate_distance(click_lat, click_lon, info["lat"], info["lon"])
        if distance < min_distance:
            min_distance = distance
            closest_location = loc

    if closest_location:
        # เลือกข้อมูล historical_data ตาม time_frame
        if time_frame == "6H":
            historical_data = historical_data_6H
        else:
            historical_data = historical_data_1D

        # ดึงข้อมูลล่าสุดจาก historical_data
        latest_data = historical_data[closest_location].iloc[-1]
        pm25 = latest_data["pm_2_5"]  # ดึงค่า PM2.5 ล่าสุด

        # สร้าง Popup
        popup_content = html.Div(
            [
                html.H4(f"Location: {locations[closest_location]['name']}"),
                html.P(f"PM2.5: {pm25}"),  # แสดงเฉพาะค่า PM2.5
            ]
        )

        # สร้าง Popup บนแผนที่
        return dl.Popup(
            children=popup_content,
            position=[click_lat, click_lon],
        )

    return None


# Callback สำหรับการทำนายข้อมูล
@app.callback(
    Output("prediction-modal", "is_open"),
    Output("prediction-graph", "figure"),
    Input("predict-button", "n_clicks"),
    Input("close-button", "n_clicks"),
    State("time_frame", "value"),
    State("locations", "value"),
    State("chart_type", "value"),
    State("prediction-modal", "is_open"),
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

    # ตรวจสอบว่าเป็นการกดปุ่ม Predict หรือ Close
    ctx = callback_context
    if not ctx.triggered:
        return False, {}
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "predict-button":
        if not selected_locations:
            return False, {}

        predictions = []
        for loc in selected_locations:
            # ทำนายข้อมูล
            model = models[time_frame][loc]
            X_feature = X_features[time_frame][loc]
            try:
                # แยก features (X) และ target (y)
                X = X_feature  # แยกคอลัมน์ pm_2_5 ออก
                prediction = predict_model(model, X=X)
                predictions.append(
                    {
                        "location": loc,
                        "data": prediction[
                            "y_pred"
                        ],  # PyCaret ใช้ "Label" เป็นคอลัมน์ผลลัพธ์
                        "timestamps": X_feature.index.to_timestamp(),
                    }
                )
            except Exception as e:
                print(f"Error predicting for {loc}: {e}")
                continue

        # สร้างกราฟ
        if chart_type == "line":
            fig = px.line()
            for pred in predictions:
                # เพิ่มข้อมูล 7 วันย้อนหลัง
                if time_frame == "6H":
                    historical = historical_data_6H[pred["location"]]
                else:
                    historical = historical_data_1D[pred["location"]]

                fig.add_scatter(
                    x=historical.index.to_timestamp(),
                    y=historical["pm_2_5"],
                    name=f"{locations[pred['location']]['name']} (Historical)",
                    mode="lines",
                    line=dict(dash="dash"),
                )

                # เพิ่มข้อมูลที่ทำนาย
                fig.add_scatter(
                    x=pred["timestamps"],
                    y=pred["data"],
                    name=f"{locations[pred['location']]['name']} (Predicted)",
                    mode="lines",
                )
        else:
            fig = px.bar()
            for pred in predictions:
                # เพิ่มข้อมูล 7 วันย้อนหลัง
                if time_frame == "6H":
                    historical = historical_data_6H[pred["location"]]
                else:
                    historical = historical_data_1D[pred["location"]]

                fig.add_bar(
                    x=historical.index.to_timestamp(),
                    y=historical["pm_2_5"],
                    name=f"{locations[pred['location']]['name']} (Historical)",
                )

                # เพิ่มข้อมูลที่ทำนาย
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

        return True, fig  # เปิด Modal และแสดงกราฟ

    elif button_id == "close-button":
        return False, {}  # ปิด Modal

    return False, {}


if __name__ == "__main__":
    app.run_server(debug=True)
