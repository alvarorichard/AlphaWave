import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

# Configurar yfinance para baixar dados
yf.pdr_override()

# Baixar dados mais recentes disponíveis da Apple
ticker = "AAPL"
data = yf.download(ticker, start="2000-01-01")

# Calcular a média anual do preço de fechamento
data['Year'] = data.index.year
annual_mean = data.groupby('Year').mean()['Close']

# Obter e imprimir o valor mais recente da ação
most_recent_value = data['Close'].iloc[-1]
current_year = datetime.now().year
print(f"Preço atual da ação hoje: ${most_recent_value:.2f}")

# Atualizar o valor mais recente para o ano atual (2023)
annual_mean.loc[current_year] = most_recent_value

# Preparar dados para ARIMA
y = annual_mean.values

# Treinar o modelo ARIMA
model = ARIMA(y, order=(1,1,1))
model_fit = model.fit()

# Fazer previsões para os anos de 2023 a 2030
years_to_predict = 8  # De 2023 a 2030
forecast = model_fit.forecast(steps=years_to_predict)

# Mostrar previsões no terminal
for i, year in enumerate(range(current_year, current_year + years_to_predict)):
    print(f"Previsão para {year}: ${forecast[i]:.2f}")

# Plotar os resultados
plt.figure(figsize=(14, 7))
plt.plot(annual_mean.index, annual_mean.values, label='Média Anual Real', marker='o', color='blue')
plt.plot(range(current_year, current_year + years_to_predict), forecast, label='Previsão ARIMA', linestyle='dashed', marker='x', color='red')

# Anotar valores no gráfico
for i, year in enumerate(range(current_year, current_year + years_to_predict)):
    plt.annotate(f"${forecast[i]:.2f}", (year, forecast[i]), textcoords="offset points", xytext=(0,5), ha='center')

plt.title('Previsão de Preços Médios Anuais de Ações da Apple')
plt.xlabel('Ano')
plt.ylabel('Preço Médio Anual ($)')
plt.legend()
plt.grid(True)

# Salvar o gráfico em um arquivo PNG
filename = "AAPL_annual_stock_price_ARIMA_prediction.png"
plt.savefig(filename, dpi=300)
print(f"\nGráfico salvo como {filename}")

plt.show()
