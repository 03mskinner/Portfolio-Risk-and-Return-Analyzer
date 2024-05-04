# %%
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash import Dash, html, dcc, dash_table, Input, Output, State, ALL, no_update
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
from dash import Dash, dcc, html, Input, Output, State, callback_context
import json

# Function to fetch historical data and financial ratios for a list of stock tickers over a specified date range
def fetch_data(stocks, start_date, end_date):
    stock_data = {}
    financials = {}
    for stock in stocks:
        try:
            data = yf.download(stock, start=start_date, end=end_date)
            if not data.empty:
                stock_data[stock] = data['Adj Close']
                ticker = yf.Ticker(stock)
                info = ticker.info
                financials[stock] = {
                    'debt_to_equity': info.get('debtToEquity', 'N/A'),
                    'pe_ratio': info.get('forwardPE', 'N/A'),
                    'peg_ratio': info.get('pegRatio', 'N/A'),
                    'roe': info.get('returnOnEquity', 'N/A')
                }
        except Exception as e:
            print(f"Failed to fetch data for {stock}: {e}")
    return pd.DataFrame(stock_data), financials


# Function to calculate returns and volatility
def calculate_metrics(stock_df, ticker):
    if isinstance(stock_df, pd.Series):  # If the returned object is a Series
        stock_df = stock_df.to_frame().T  # Transpose to make it a single row DataFrame
        stock_df.columns = [ticker]  # Rename the column to the stock ticker
    
    returns = stock_df.pct_change().dropna()
    avg_returns = returns.mean() * 252  # Annual returns
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility

    return avg_returns, volatility



# Function to create a graph for the portfolio
def create_portfolio_graph(stock_df, weights):
    weighted_stock_df = stock_df.multiply(weights, axis='columns').sum(axis=1)
    trace = go.Scatter(x=weighted_stock_df.index, y=weighted_stock_df, mode='lines', name='Portfolio')
    layout = go.Layout(title='Portfolio Value Over Time', xaxis={'title': 'Date'}, yaxis={'title': 'Portfolio Value'})
    fig = go.Figure(data=[trace], layout=layout)
    return fig

# Function to create comparative graphs for stocks
def create_individual_stock_graph(stock_df):
    traces = []
    for col in stock_df.columns:
        traces.append(go.Scatter(x=stock_df.index, y=stock_df[col], mode='lines', name=col))
    layout = go.Layout(title='Stock Prices Over Time', xaxis={'title': 'Date'}, yaxis={'title': 'Adjusted Close Price'})
    fig = go.Figure(data=traces, layout=layout)
    return fig

# Setting up Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H3("Portfolio Input"),
        dcc.Input(id='input-start-date', type='text', placeholder='Start Date (YYYY-MM-DD)', value='2020-01-01'),
        dcc.Input(id='input-end-date', type='text', placeholder='End Date (YYYY-MM-DD)', value='2024-12-31'),
        html.Button('Add Stock', id='add-stock-button', n_clicks=0),
        html.Button('Submit Portfolio', id='submit-portfolio-button', n_clicks=0),
        html.Button('Analyze Individual Stock', id='analyze-stock-button', n_clicks=0),
        dcc.Input(id='individual-stock-ticker', type='text', placeholder='Individual stock ticker...'),
        dcc.Input(id='desired-return', type='number', placeholder='Desired Return %'),
        dcc.Input(id='max-risk', type='number', placeholder='Maximum Risk %')
    ], style={'padding': '20px'}),
    html.Div(id='stock-inputs', children=[]),
    html.Div(id='output-portfolio-graph'),
    html.Div(id='output-portfolio-stats'),
    html.Div(id='output-individual-stock-analysis'),
    html.Div(id='comparison-output', children=[
        dcc.Graph(id='comparison-graph', figure=go.Figure(), style={'display': 'none'}),  # Initially hidden
        html.Div(id='comparison-stats', style={'display': 'none'})  # Initially hidden
    ])
])

@app.callback(
    Output('stock-inputs', 'children'),
    [Input('add-stock-button', 'n_clicks'),
     Input({'type': 'remove-stock-button', 'index': ALL}, 'n_clicks')],
    [State('stock-inputs', 'children'),
     State({'type': 'remove-stock-button', 'index': ALL}, 'n_clicks_timestamp')],
    prevent_initial_call=True
)
def update_stock_inputs(add_clicks, remove_clicks, children, timestamps):
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if 'add-stock-button' in trigger_id:
        index = len(children)
        new_element = html.Div([
            dcc.Input(type='text', placeholder='Enter stock ticker...'),
            dcc.Input(type='number', placeholder='Weight %', min=0, max=100, step=1),
            html.Button('Remove Stock', id={'type': 'remove-stock-button', 'index': index})
        ], style={'margin': '10px'})
        children.append(new_element)
    elif 'remove-stock-button' in trigger_id:
        index_to_remove = json.loads(trigger_id)['index']
        children.pop(index_to_remove)

    return children


@app.callback(
    [Output('output-portfolio-graph', 'children'),
     Output('output-portfolio-stats', 'children')],
    Input('submit-portfolio-button', 'n_clicks'),
    [State('stock-inputs', 'children'),
     State('input-start-date', 'value'),
     State('input-end-date', 'value')],
    prevent_initial_call=True
)
def update_portfolio_output(n_clicks, inputs, start_date, end_date):
    if not inputs:
        return '', 'Please add at least one stock to the portfolio.'

    stocks = [input['props']['children'][0]['props']['value'].upper() for input in inputs]
    weights = [float(input['props']['children'][1]['props']['value']) for input in inputs]

    stock_data, financials = fetch_data(stocks, start_date, end_date)
    if stock_data.empty:
        return '', 'No data available for the selected stocks.'

    # Initialize Series for avg returns and volatility to store results
    avg_returns = pd.Series(index=stocks)
    volatility = pd.Series(index=stocks)

    # Calculate metrics for each stock and store in Series
    for stock in stocks:
        if stock in stock_data.columns:
            single_stock_data = stock_data[[stock]].dropna()
            stock_avg_returns, stock_volatility = calculate_metrics(single_stock_data, stock)
            avg_returns[stock] = stock_avg_returns.iloc[0]
            volatility[stock] = stock_volatility.iloc[0]

    # Generate table data
    table_data = [{
        'Stock': stock,
        'Weight': f"{weights[idx]*100}%",
        'Avg Return': f"{avg_returns[stock]:.2%}",
        'Volatility': f"{volatility[stock]:.2%}",
        **financials.get(stock, {})
    } for idx, stock in enumerate(stocks)]

    table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in table_data[0].keys()],
        data=table_data,
        style_table={'overflowX': 'auto'}
    )

    fig = create_portfolio_graph(stock_data, weights)
    portfolio_stats = "Portfolio Analysis Complete"
    return dcc.Graph(figure=fig), html.Div([table, html.P(portfolio_stats)])



def compare_portfolio_and_stock(portfolio_data, stock_data, stock_ticker):
    # Calculate metrics for the stock
    stock_returns = stock_data.pct_change().dropna()
    stock_avg_return = stock_returns.mean() * 252
    stock_volatility = stock_returns.std() * np.sqrt(252)
    
    # Prepare data for the graph
    portfolio_val = portfolio_data.sum(axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_val.index, y=portfolio_val, mode='lines', name='Portfolio'))
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data[stock_ticker], mode='lines', name=stock_ticker))
    fig.update_layout(title='Portfolio vs Stock Comparison', xaxis_title='Date', yaxis_title='Value')
    
    # Prepare statistical summary
    stats = {
        'Portfolio Avg Return': f"{portfolio_val.pct_change().mean() * 252:.2%}",
        'Portfolio Volatility': f"{portfolio_val.pct_change().std() * np.sqrt(252):.2%}",
        'Stock Avg Return': f"{stock_avg_return:.2%}",
        'Stock Volatility': f"{stock_volatility:.2%}"
    }
    
    return fig, stats

@app.callback(
    Output('output-individual-stock-analysis', 'children'),
    Input('analyze-stock-button', 'n_clicks'),
    State('individual-stock-ticker', 'value'),
    State('input-start-date', 'value'),
    State('input-end-date', 'value'),
    prevent_initial_call=True
)
def analyze_individual_stock(n_clicks, stock, start_date, end_date):
    if n_clicks == 0 or not stock:
        return html.Div("Please enter a stock ticker.")

    stock = stock.strip().upper()
    stock_data, financials = fetch_data([stock], start_date, end_date)
    if stock_data.empty or stock not in stock_data.columns:
        return html.Div("No data available for the given stock ticker.")

    single_stock_data = stock_data[[stock]].dropna()
    avg_returns, volatility = calculate_metrics(single_stock_data, stock)
    fig = create_individual_stock_graph(single_stock_data)

    stats = f"Individual Stock Analysis for {stock}: Avg Return = {avg_returns.iloc[0]:.2%}, Volatility = {volatility.iloc[0]:.2%}, Debt to Equity = {financials[stock]['debt_to_equity']}, PE Ratio = {financials[stock]['pe_ratio']}, PEG Ratio = {financials[stock]['peg_ratio']}, ROE = {financials[stock]['roe']}"
    return html.Div([dcc.Graph(figure=fig), html.P(stats)])


import socket

# Function to find a free port for the server
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

if __name__ == '__main__':
    port = find_free_port()
    web = f"http://127.0.0.1:{port}"
    print(web)
    app.run_server(debug=True, port=port)

# %%
