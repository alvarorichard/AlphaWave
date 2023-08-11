import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Configurar yfinance para baixar dados
yf.pdr_override()

# Baixar dados mais recentes disponíveis da Apple
ticker = "AAPL"
data = yf.download(ticker, start="2000-01-01")

# Calcular a média anual do preço de fechamento
data['Year'] = data.index.year
annual_mean = data.groupby('Year').mean()['Close']

# Preparar dados para regressão
X = annual_mean.index.values.reshape(-1, 1)
y = annual_mean.values

# Treinar o modelo de regressão linear
model = LinearRegression()
model.fit(X, y)

# Fazer previsões para os próximos anos
years_to_predict = list(range(2000, 2031))  # Previsão até 2030
predictions = model.predict(np.array(years_to_predict).reshape(-1, 1))

# Mostrar previsões no terminal
for year, prediction in zip(years_to_predict, predictions):
    print(f"Previsão para {year}: ${prediction:.2f}")

# Plotar os resultados
plt.figure(figsize=(14, 7))
plt.plot(annual_mean.index, annual_mean.values, label='Média Anual Real', marker='o', color='blue')
plt.plot(years_to_predict, predictions, label='Previsão Regressão Linear', linestyle='dashed', color='red')

# Anotar valores no gráfico
for year, prediction in zip(years_to_predict, predictions):
    plt.annotate(f"${prediction:.2f}", (year, prediction), textcoords="offset points", xytext=(0,5), ha='center')

plt.title('Previsão de Preços Médios Anuais de Ações da Apple')
plt.xlabel('Ano')
plt.ylabel('Preço Médio Anual ($)')
plt.legend()
plt.grid(True)

# Salvar o gráfico em um arquivo PNG
filename = "AAPL_annual_stock_price_prediction.png"
plt.savefig(filename, dpi=300)
print(f"\nGráfico salvo como {filename}")

plt.show()
