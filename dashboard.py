from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

df = pd.read_csv('C:\\0_DA\\Iot_DataAnalyst\\clustering_results.csv')

df.columns = df.columns.str.strip()

if 'Power Factor' in df.columns:
    df.rename(columns={'Power Factor': 'PowerFactor'}, inplace=True)

df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

df["Hour"] = df["Timestamp"].dt.hour
df["Weekday"] = df["Timestamp"].dt.day_name()
df["Month"] = df["Timestamp"].dt.month

df = df.dropna(subset=["Month", "Hour", "Weekday", "Voltage (V)", "Power Consumption (kW)"])

df["Month"] = df["Month"].astype(int)

app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

app.layout = dbc.Container(fluid=True, children=[

    dbc.Row([
        dbc.Col(html.H1("Smart Energy IoT Dashboard",
                        className="text-center mt-4 mb-4"), width=12)
    ]),

    dbc.Tabs([

        dbc.Tab(label="RQ1: Consumption Patterns", children=[
            html.Br(),

            dcc.Dropdown(
                id="rq1-agg",
                options=[
                    {"label": "Hourly", "value": "Hour"},
                    {"label": "Weekly", "value": "Weekday"},
                    {"label": "Monthly", "value": "Month"},
                ],
                value="Hour",
                clearable=False,
                style={"width": "40%"}
            ),

            dcc.Graph(id="rq1-plot"),
        ]),

        dbc.Tab(label="RQ2: Power & Quality Relations", children=[
            html.Br(),

            dcc.Dropdown(
                id="rq2-feature",
                options=[
                    {"label": "Voltage (V)", "value": "Voltage (V)"},
                    {"label": "Reactive Power (kVAR)", "value": "Reactive Power (kVAR)"},
                    {"label": "Power Factor", "value": "PowerFactor"},
                ],
                value="Voltage (V)",
                clearable=False,
                style={"width": "40%"}
            ),

            dcc.Graph(id="rq2-scatter"),
        ]),

        dbc.Tab(label="RQ3: Clustering & Anomalies", children=[
            html.Br(),

            dcc.Dropdown(
                id="rq3-model",
                options=[
                    {"label": "KMeans", "value": "Cluster_KMeans"},
                    {"label": "Hierarchical", "value": "Cluster_Hierarchical"},
                    {"label": "DBSCAN", "value": "Cluster_DBSCAN"},
                    {"label": "GMM", "value": "Cluster_GMM"},
                ],
                value="Cluster_KMeans",
                clearable=False,
                style={"width": "40%"}
            ),

            dcc.Graph(id="rq3-scatter"),
            dcc.Graph(id="rq3-timeseries")
        ]),

        dbc.Tab(label="RQ4: Geographic Map", children=[
            html.Br(),

            dcc.Dropdown(
                id="rq4-metric",
                options=[
                    {"label": "Power Consumption (kW)", "value": "Power Consumption (kW)"},
                    {"label": "Voltage (V)", "value": "Voltage (V)"},
                    {"label": "Power Factor", "value": "PowerFactor"},
                    {"label": "Temperature (°C)", "value": "Temperature (°C)"},
                ],
                value="Power Consumption (kW)",
                clearable=False,
                style={"width": "40%"}
            ),

            dcc.Graph(id="rq4-map"),
        ]),

        dbc.Tab(label="RQ5: Monthly Evolution", children=[
            html.Br(),

            dcc.Dropdown(
                id="rq5-x-axis",
                options=[
                    {"label": "Voltage (V)", "value": "Voltage (V)"},
                    {"label": "Current (A)", "value": "Current (A)"},
                    {"label": "Temperature (°C)", "value": "Temperature (°C)"},
                ],
                value="Voltage (V)",
                clearable=False,
                style={"width": "40%", "display": "inline-block"}
            ),

            dcc.Dropdown(
                id="rq5-y-axis",
                options=[
                    {"label": "Power Consumption (kW)", "value": "Power Consumption (kW)"},
                    {"label": "Power Factor", "value": "PowerFactor"},
                    {"label": "Reactive Power (kVAR)", "value": "Reactive Power (kVAR)"},
                ],
                value="Power Consumption (kW)",
                clearable=False,
                style={"width": "40%", "display": "inline-block", "marginLeft": "2%"}
            ),

            dcc.Graph(id="rq5-animated-scatter"),
        ]),
    ])
])


@app.callback(
    Output("rq1-plot", "figure"),
    Input("rq1-agg", "value")
)
def update_rq1(agg):
    try:
        if agg == "Month":
            df_sample = df.sample(min(3000, len(df))).copy()

            city_columns = [col for col in df_sample.columns if col.startswith('City_')]
            if len(city_columns) > 0:
                df_sample['City'] = df_sample[city_columns].idxmax(axis=1).str.replace('City_', '')
            else:
                df_sample['City'] = 'Unknown'

            df_sample = df_sample.sort_values('Month')

            fig = px.scatter(
                df_sample,
                x="Voltage (V)",
                y="Power Consumption (kW)",
                animation_frame="Month",
                color="City",
                size="Power Consumption (kW)",
                hover_name="City",
                title="Power Consumption vs Voltage (Monthly Evolution)",
                height=600,
                category_orders={"Month": sorted(df_sample['Month'].unique())}
            )

            fig.update_layout(
                xaxis_title="Voltage (V)",
                yaxis_title="Power Consumption (kW)",
                showlegend=True
            )
        else:
            agg_data = df.groupby(agg)["Power Consumption (kW)"].mean().reset_index()
            fig = px.line(
                agg_data,
                x=agg,
                y="Power Consumption (kW)",
                title=f"Average Power Consumption by {agg}"
            )

        return fig
    except Exception as e:
        print(f"Error in RQ1: {e}")
        return px.line(title="Error loading chart")


@app.callback(
    Output("rq2-scatter", "figure"),
    Input("rq2-feature", "value")
)
def update_rq2(feature):
    try:
        sample_data = df.sample(min(3000, len(df)))
        fig = px.scatter(
            sample_data,
            x="Power Consumption (kW)",
            y=feature,
            color="Month",
            title=f"Power Consumption vs {feature}"
        )
        return fig
    except Exception as e:
        print(f"Error in RQ2: {e}")
        return px.scatter(title="Error loading chart")


@app.callback(
    Output("rq3-scatter", "figure"),
    Output("rq3-timeseries", "figure"),
    Input("rq3-model", "value")
)
def update_rq3(model):
    try:
        sample_data = df.sample(min(3000, len(df)))

        fig1 = px.scatter(
            sample_data,
            x="Voltage (V)",
            y="Power Consumption (kW)",
            color=model,
            title=f"Clustering Visualization using Voltage vs Consumption ({model})",
            opacity=0.8
        )

        cluster_value = df[model].unique()[0]
        df_c = df[df[model] == cluster_value]

        fig2 = px.line(
            df_c.head(2000),
            x="Timestamp",
            y="Power Consumption (kW)",
            title=f"Time-Series for {model} - Cluster {cluster_value}",
            markers=False
        )

        return fig1, fig2
    except Exception as e:
        print(f"Error in RQ3: {e}")
        empty_fig = px.scatter(title="Error loading chart")
        return empty_fig, empty_fig


@app.callback(
    Output("rq4-map", "figure"),
    Input("rq4-metric", "value")
)
def update_rq4_map(metric):
    try:
        city_country_map = {
            'Bangkok': 'THA', 'Beijing': 'CHN', 'Delhi': 'IND',
            'Dhaka': 'BGD', 'Hanoi': 'VNM', 'Jakarta': 'IDN',
            'Karachi': 'PAK', 'KualaLumpur': 'MYS', 'Manila': 'PHL',
            'Mumbai': 'IND', 'Seoul': 'KOR', 'Shanghai': 'CHN',
            'Singapore': 'SGP', 'Tashkent': 'UZB', 'Tokyo': 'JPN'
        }

        city_columns = [col for col in df.columns if col.startswith('City_')]

        city_data = []
        for city_col in city_columns:
            city_name = city_col.replace('City_', '')
            if city_name in city_country_map:
                avg_value = df[df[city_col] == 1][metric].mean()
                if not pd.isna(avg_value):
                    city_data.append({
                        'City': city_name,
                        'Country': city_country_map[city_name],
                        'Value': avg_value
                    })

        df_map = pd.DataFrame(city_data)

        fig = px.choropleth(
            df_map,
            locations="Country",
            locationmode="ISO-3",
            color="Value",
            hover_name="City",
            hover_data={"Country": True, "Value": ":.2f"},
            color_continuous_scale="Viridis",
            title=f"Average {metric} by Country",
            projection="natural earth"
        )

        fig.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth'
            )
        )

        return fig
    except Exception as e:
        print(f"Error in RQ4: {e}")
        return px.choropleth(title="Error loading map")


@app.callback(
    Output("rq5-animated-scatter", "figure"),
    Input("rq5-x-axis", "value"),
    Input("rq5-y-axis", "value")
)
def update_rq5_animated(x_axis, y_axis):
    try:
        df_sample = df.sample(min(3000, len(df))).copy()

        city_columns = [col for col in df_sample.columns if col.startswith('City_')]
        if len(city_columns) > 0:
            df_sample['City'] = df_sample[city_columns].idxmax(axis=1).str.replace('City_', '')
        else:
            df_sample['City'] = 'Unknown'

        df_sample = df_sample.sort_values('Month')

        fig = px.scatter(
            df_sample,
            x=x_axis,
            y=y_axis,
            animation_frame="Month",
            color="City",
            size="Power Consumption (kW)",
            hover_name="City",
            title=f"{y_axis} vs {x_axis} (Monthly Evolution)",
            height=600,
            category_orders={"Month": sorted(df_sample['Month'].unique())}
        )

        fig.update_layout(
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            showlegend=True
        )

        return fig
    except Exception as e:
        print(f"Error in RQ5: {e}")
        return px.scatter(title="Error loading chart")


if __name__ == "__main__":
    app.run(debug=True)
