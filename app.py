import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from io import BytesIO
import os
from dotenv import load_dotenv

# === Load Environment Variables ===
load_dotenv()

# === PostgreSQL connection setup ===
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

# === Streamlit App ===
st.title("Market Data Dashboard (ML, Insights & Alerts)")
st.markdown("_Disclaimer: This dashboard is for informational and educational purposes only. It does not constitute financial advice._")

symbol = st.text_input("Enter a stock symbol (e.g., AAPL, TSLA):", "AAPL")
period = st.selectbox("Select time period:", ["1mo", "3mo", "6mo", "1y"])

if symbol:
    stock = yf.Ticker(symbol)
    try:
        hist_raw = stock.history(period=period)
    except Exception as e:
        st.error("Error fetching stock data. Please check your internet connection or the symbol.")
        st.stop()

    if not hist_raw.empty:
        hist = hist_raw.copy()
        hist['date'] = hist.index.date
        hist = hist.groupby('date').agg({
            'Close': 'last',
            'Volume': 'last'
        }).reset_index()

        # === Chart Section ===
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist['date'], y=hist['Close'], name='Close Price', line=dict(color='blue')))
        fig.add_trace(go.Bar(x=hist['date'], y=hist['Volume'], name='Volume', marker=dict(color='lightgray'),
                             yaxis='y2', opacity=0.4))
        fig.update_layout(
            title=f"{symbol.upper()} - Price & Volume",
            xaxis_title="Date",
            yaxis=dict(title='Price'),
            yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # === Technical Indicators ===
        df = hist[['date', 'Close', 'Volume']].copy()
        df.rename(columns={'Close': 'close'}, inplace=True)
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA30'] = df['close'].rolling(window=30).mean()
        df['return_1d'] = df['close'].pct_change()
        df['momentum_5d'] = df['close'] - df['close'].shift(5)
        df['vol_change'] = df['Volume'].pct_change()
        df['volatility_5d'] = df['close'].rolling(window=5).std()
        df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
        delta = df['close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean()
        avg_loss = pd.Series(loss).rolling(14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
        upper = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
        lower = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
        df['BB_Width'] = (upper - lower) / df['close'].rolling(20).mean()
        df['ROC'] = df['close'].pct_change(periods=5)

        df = df.dropna()

        # === ML Prediction ===
        st.subheader("Predict Tomorrow's Closing Price")
        st.caption("_Note: This prediction is for educational purposes only and should not be considered financial or investment advice._")
        features = ['MA5', 'MA10', 'MA30', 'return_1d', 'momentum_5d', 'vol_change', 'volatility_5d',
                    'EMA20', 'RSI', 'MACD', 'BB_Width', 'ROC']
        X = df[features]
        y = df['close']

        if len(X) > 0:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            df['predicted'] = model.predict(X)
            predicted_close = df['predicted'].iloc[-1]
            current_close = df['close'].iloc[-1]

            st.success(f"Predicted Close for Tomorrow: **${predicted_close:.2f}**")

            if predicted_close > current_close * 1.05:
                st.info("**BUY SIGNAL:** Prediction is more than 5% above current close!")

            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=df['date'], y=df['close'], name='Actual'))
            fig_pred.add_trace(go.Scatter(x=df['date'], y=df['predicted'], name='Predicted'))
            fig_pred.update_layout(title='Actual vs Predicted Close', xaxis_title='Date', yaxis_title='Price')
            st.plotly_chart(fig_pred)

            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)
            fig_imp, ax = plt.subplots()
            ax.barh(range(len(features)), importances[sorted_idx])
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(np.array(features)[sorted_idx])
            ax.set_xlabel("Importance")
            ax.set_title("Feature Importance")
            st.pyplot(fig_imp)

            st.subheader("Export Prediction Data")
            export_df = df[['date', 'close', 'predicted']].copy()
            csv = export_df.to_csv(index=False)
            st.download_button("Download as CSV", csv, file_name=f"{symbol}_predictions.csv", mime='text/csv')

            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                export_df.to_excel(writer, index=False, sheet_name='Predictions')
            st.download_button("Download as Excel", excel_buffer.getvalue(), file_name=f"{symbol}_predictions.xlsx")

            # === Backtesting Integration ===
            st.subheader("Backtest Strategy Using Predictions")
            st.caption(
    "This simulation assumes you start with $10,000 and only buy when the model predicts tomorrow's price "
    "will be more than the current price by the selected threshold (e.g., {threshold}%). "
    "It sells when the predicted price is lower. This helps visualize potential strategy performance based on the model.")
            threshold = st.slider("Set Buy Signal Threshold (% above current price):", min_value=1, max_value=10, value=5)

            def backtest_strategy(df, initial_cash=10000, threshold_pct=5):
                cash = initial_cash
                shares = 0
                trades = 0
                history = []
                for i in range(len(df)):
                    price = df.iloc[i]['close']
                    pred = df.iloc[i]['predicted']
                    date = df.iloc[i]['date']
                    if pred > price * (1 + threshold_pct / 100) and shares == 0:
                        shares = int(cash // price)
                        cash -= shares * price
                        trades += 1
                    elif pred < price and shares > 0:
                        cash += shares * price
                        shares = 0
                        trades += 1
                    portfolio_value = cash + shares * price
                    history.append({'date': date, 'portfolio_value': portfolio_value})
                final_value = history[-1]['portfolio_value']
                return_pct = (final_value - initial_cash) / initial_cash * 100
                return final_value, return_pct, trades, pd.DataFrame(history)

            final_val, return_pct, trades, backtest_df = backtest_strategy(df, threshold_pct=threshold)

            st.write(f"**Final Portfolio Value:** ${final_val:,.2f}")
            st.write(f"**Total Return:** {return_pct:.2f}%")
            st.write(f"**Number of Trades:** {trades}")
            st.line_chart(backtest_df.set_index('date')['portfolio_value'])

        else:
            st.warning("Not enough data to train model. Try a longer time range.")

        # === Save to PostgreSQL ===
        hist['symbol'] = symbol.upper()
        hist.rename(columns={'Close': 'close'}, inplace=True)
        try:
            with engine.begin() as conn:
                for _, row in hist.iterrows():
                    stmt = text("""
                        INSERT INTO stock_data (symbol, date, close)
                        VALUES (:symbol, :date, :close)
                        ON CONFLICT (symbol, date)
                        DO UPDATE SET close = EXCLUDED.close;
                    """)
                    conn.execute(stmt, {
                        "symbol": row['symbol'],
                        "date": row['date'],
                        "close": row['close']
                    })
        except Exception as e:
            st.warning("Database update skipped (connection error or not available).")

        # === View Stored Data ===
        st.subheader(f"Stored Data for {symbol.upper()}")
        try:
            start = hist['date'].min()
            end = hist['date'].max()
            query = f"""
                SELECT * FROM stock_data
                WHERE symbol = '{symbol.upper()}'
                AND date BETWEEN '{start}' AND '{end}'
                ORDER BY date DESC
            """
            filtered_df = pd.read_sql(query, engine)
            st.dataframe(filtered_df)
        except Exception as e:
            st.warning("Could not fetch stored data.")
    else:
        st.warning("No data found.")
