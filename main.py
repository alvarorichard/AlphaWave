import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
import json
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime


# Configurar yfinance para baixar dados
yf.pdr_override()
nltk.download('vader_lexicon')

# Função para obter dados da News API e analisar o sentimento usando VADER
def get_sentiment(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey=56e8609d24dc45328e7968ae21e0ebad"
    response = requests.get(url)
    news_data = json.loads(response.text)
    articles = news_data['articles']

    sia = SentimentIntensityAnalyzer()

    sentiment_score = 0
    for article in articles:
        sentiment = sia.polarity_scores(article['title'])
        sentiment_score += sentiment['compound']

    average_sentiment = sentiment_score / len(articles) if articles else 0

    return average_sentiment


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
    sentiment = get_sentiment(ticker)  # Obter dados de sentimento

    # Imprimir previsões no terminal
    for i, year in enumerate(range(current_year, current_year + years_to_predict)):
        print(f"Previsão para {ticker} em {year}: ${forecast[i]:.2f}")

    print(f"Sentimento médio para {ticker}: {sentiment}")

    return annual_mean, forecast, sentiment


# Baixar dados e fazer previsões para Apple e Nestlé
tickers = ['AAPL', 'NSRGY']
predictions = {}

for ticker in tickers:
    annual_mean, forecast, sentiment = get_forecast(ticker)
    predictions[ticker] = {'annual_mean': annual_mean, 'forecast': forecast, 'sentiment': sentiment}

# Plotar os resultados
plt.figure(figsize=(14, 7))

for ticker in tickers:
    plt.plot(predictions[ticker]['annual_mean'].index, predictions[ticker]['annual_mean'].values,
             label=f'Média Anual Real de {ticker}', marker='o')
    current_year = datetime.now().year
    plt.plot(range(current_year, current_year + 8), predictions[ticker]['forecast'],
             label=f'Previsão ARIMA de {ticker}', linestyle='dashed', marker='x')
    plt.axhline(y=predictions[ticker]['sentiment'], color='r', linestyle='-',
                label=f"Sentimento Médio de {ticker}")

plt.title('Previsão e Análise de Sentimento de Preços Médios Anuais de Ações da Apple e Nestlé')
plt.xlabel('Ano')
plt.ylabel('Preço Médio Anual ($)')
plt.legend()
plt.grid(True)

# Salvar o gráfico em um arquivo PNG
filename = "AAPL_NSRGY_annual_stock_price_ARIMA_sentiment_prediction.png"
plt.savefig(filename, dpi=300)
print(f"\nGráfico salvo como {filename}")

plt.show()
