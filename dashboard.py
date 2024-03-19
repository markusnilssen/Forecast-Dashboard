# dash.plotly.com

import pandas as pd
import plotly.express as px
import pickle
import matplotlib.pyplot as plt
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np

app = dash.Dash(__name__)

#--------------------------------------------------------
# Import data and clean data
df_raw = pd.read_csv('testData_2019_SouthTower.csv')
for key in df_raw.iloc[:,1:]:
    df_raw[key] = df_raw[key].round(2)

df_clean = df_raw.copy()
df_clean['Date'] = pd.to_datetime(df_clean['Date'])
df_clean = df_clean.set_index('Date', drop=True)

# Import model and the data that has gone through festure selection in project 1
df_model = pd.read_csv('2019_test.csv')
df_model['Date'] = pd.to_datetime(df_model['Date'])
df_model = df_model.set_index('Date',drop=True)

with open('xgb_model.pkl','rb') as file:
    xgb_model = pickle.load(file)

start_date = '2019-01-01'
end_date = '2019-03-31'
filtered_df = df_model[start_date:end_date]

Z = filtered_df.values
Y = Z[:,0]
X2 = Z[:,[1,2,3]]

y_pred = xgb_model.predict(X2)

fig = {
    'data': [
        go.Scatter(
            x=filtered_df.index,
            y=Y,
            mode='lines+markers',
            name='Actuall power values'
        ),
        go.Scatter(
            x=filtered_df.index,
            y=y_pred,
            mode='lines+markers',
            name='Predicted power values'
        )
    ],
    'layout': go.Layout(
        xaxis={'title': 'Date'},
        yaxis={'title': 'Power (kW)'}
    )
}

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Calculate Mean Squared Error (MSE)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

mae_2019 = mean_absolute_error(Y, y_pred)
# Mean Squared Error (MSE)
mse_2019 = mean_squared_error(Y, y_pred)
# Root Mean Squared Error (RMSE)
rmse_2019 = np.sqrt(mse_2019)
# Mean Bias Error (MBE)
mbe_2019 = np.mean(Y - y_pred)
# Relative Mean Bias Error (RMBE)
rmbe_2019 = mbe_2019 / np.mean(Y)
# Coefficient of Variation Root Mean Squared Error (cvRMSE)
cvrmse_2019 = rmse_2019 / np.mean(Y)
data_dict = dict()
values = [mae_2019.round(3),mse_2019.round(3),rmse_2019.round(3),mbe_2019.round(3),rmbe_2019.round(3),cvrmse_2019.round(3)]
metrics = ['MAE','MSE','RMSE','MBE','RMBE','cvRMSE']
#--------------------------------------------------------
# App layout

app.layout = html.Div([
    html.H1('Power forecast dashboard'),
    dcc.Tabs(id="tabs-example-graph", value='tab-1-example-graph', children=[
        dcc.Tab(label='Data', value='tab-1'),
        dcc.Tab(label='Forecast and metrics', value='tab-2'),
    ]),
    html.Div(id='tabs-content-example-graph')  
])

#--------------------------------------------------------

@app.callback(Output('tabs-content-example-graph', 'children'),
              Input('tabs-example-graph', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Raw data 2019',style={'text-align':'center'}),
            html.Div([
                html.Div([
                    dash.dash_table.DataTable(df_raw.to_dict('records'), [{"name": i, "id": i} for i in df_raw.columns])
                ],style={'width': '70%'})
            ],style={'display':'flex','justify-content':'center','align-items':'center'})
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Forecast january to march 2019, with metrics',style={'text-align':'center'}),
            html.Div([
                dcc.Graph(figure=fig, style={'width':'80%'}),
                html.Table([
                    # Header Row
                    html.Tr([html.Th("Metric",style={'border-bottom':'1px solid black'})] + [html.Th("Value",style={'border-bottom':'1px solid black'})]),
                    # Data Rows
                    html.Tr([html.Td(metrics[0])] + [html.Th(values[0])]),
                    html.Tr([html.Td(metrics[1])] + [html.Th(values[1])]),
                    html.Tr([html.Td(metrics[2])] + [html.Th(values[2])]),
                    html.Tr([html.Td(metrics[3])] + [html.Th(values[3])]),
                    html.Tr([html.Td(metrics[4])] + [html.Th(values[4])]),
                    html.Tr([html.Td(metrics[5])] + [html.Th(values[5])]),
                ], style={'width':'19%','border':'1px solid black'})
            ], style={'display':'flex','justify-content':'center','align-items':'center'})
        ])

#--------------------------------------------------------
if __name__ == '__main__':
    app.run_server()