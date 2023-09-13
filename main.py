import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

# Configurar yfinance para baixar dados
yf.pdr_override()


# Função para baixar dados e fazer previsões
def get_forecast(ticker):
    data = yf.download(ticker, start="2000-01-01")
    data['Year'] = data.index.year
    annual_mean = data.groupby('Year').mean()['Close']
    most_recent_value = data['Close'].iloc[-1]
    current_year = datetime.now().year
    print(f"Preço atual da ação de {ticker} hoje: ${most_recent_value:.2f}")
    annual_mean.loc[current_year] = most_recent_value
    y = annual_mean.values
    model = ARIMA(y, order=(1, 1, 1))
    model_fit = model.fit()
    years_to_predict = 8  # De 2023 a 2030
    forecast = model_fit.forecast(steps=years_to_predict)

    # Imprimir previsões no terminal
    for i, year in enumerate(range(current_year, current_year + years_to_predict)):
        print(f"Previsão para {ticker} em {year}: ${forecast[i]:.2f}")

    return annual_mean, forecast


# Baixar dados e fazer previsões para Apple e Nestlé
tickers = ['AAPL', 'NSRGY']
predictions = {}
for ticker in tickers:
    annual_mean, forecast = get_forecast(ticker)
    predictions[ticker] = {'annual_mean': annual_mean, 'forecast': forecast}

# Plotar os resultados
plt.figure(figsize=(14, 7))

for ticker in tickers:
    plt.plot(predictions[ticker]['annual_mean'].index, predictions[ticker]['annual_mean'].values,
             label=f'Média Anual Real de {ticker}', marker='o')
    current_year = datetime.now().year
    plt.plot(range(current_year, current_year + 8), predictions[ticker]['forecast'],
             label=f'Previsão ARIMA de {ticker}', linestyle='dashed', marker='x')

plt.title('Previsão de Preços Médios Anuais de Ações da Apple e Nestlé')
plt.xlabel('Ano')
plt.ylabel('Preço Médio Anual ($)')
plt.legend()
plt.grid(True)

# Salvar o gráfico em um arquivo PNG
filename = "AAPL_NSRGY_annual_stock_price_ARIMA_prediction.png"
plt.savefig(filename, dpi=300)
print(f"\nGráfico salvo como {filename}")

plt.show()