import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
import json
from fpdf import FPDF
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime


# Função para ajustar a previsão com base no sentimento
def adjust_forecast_with_sentiment(forecast, sentiment):
    adjustment_factor = 1 + (sentiment * 0.1)
    adjusted_forecast = forecast * adjustment_factor
    return adjusted_forecast


# Configurar yfinance para baixar dados
yf.pdr_override()
nltk.download('vader_lexicon')




# Função para obter dados da News API e analisar o sentimento usando VADER
def get_sentiment(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}+stocks&apiKey=56e8609d24dc45328e7968ae21e0ebad"
    response = requests.get(url)
    news_data = json.loads(response.text)
    articles = news_data.get('articles', [])

    sia = SentimentIntensityAnalyzer()

    sentiment_score = 0
    count = 0
    for article in articles:
        # Filtragem de Notícias - exemplo simples, pode ser aprimorado
        if "finance" in article["source"]["name"].lower() or "business" in article["source"]["name"].lower():
            title_sentiment = sia.polarity_scores(article['title'])['compound']
            content_sentiment = sia.polarity_scores(article['description'])['compound']
            sentiment_score += title_sentiment + content_sentiment
            count += 2

    average_sentiment = sentiment_score / count if count else 0

    return average_sentiment

# ... (restante do código posterior)


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
    years_to_predict = 8
    forecast = model_fit.forecast(steps=years_to_predict)
    sentiment = get_sentiment(ticker)
    adjusted_forecast = adjust_forecast_with_sentiment(forecast, sentiment)

    # Imprimir previsões no terminal
    for i, year in enumerate(range(current_year, current_year + years_to_predict)):
        print(f"Previsão sem ajuste para {ticker} em {year}: ${forecast[i]:.2f}")
        print(f"Previsão ajustada para {ticker} em {year}: ${adjusted_forecast[i]:.2f}")

    print(f"Sentimento médio para {ticker}: {sentiment}")

    return annual_mean, forecast, adjusted_forecast, sentiment


# Baixar dados e fazer previsões para Apple
tickers = ['AAPL']
predictions = {}

for ticker in tickers:
    annual_mean, forecast, adjusted_forecast, sentiment = get_forecast(ticker)
    predictions[ticker] = {'annual_mean': annual_mean, 'forecast': forecast, 'adjusted_forecast': adjusted_forecast,
                           'sentiment': sentiment}


# Plotar os resultados
plt.figure(figsize=(14, 7))

for ticker in tickers:
    plt.plot(predictions[ticker]['annual_mean'].index, predictions[ticker]['annual_mean'].values,
             label=f'Média Anual Real de {ticker}', marker='o')
    current_year = datetime.now().year
    plt.plot(range(current_year, current_year + 8), predictions[ticker]['forecast'],
             label=f'Previsão ARIMA de {ticker}', linestyle='dashed', marker='x')
    plt.plot(range(current_year, current_year + 8), predictions[ticker]['adjusted_forecast'],
             label=f'Previsão Ajustada de {ticker}', linestyle='dotted', marker='+')
    plt.axhline(y=predictions[ticker]['sentiment'], color='r', linestyle='-',
                label=f"Sentimento Médio de {ticker}")

plt.title('Previsão e Análise de Sentimento de Preços Médios Anuais de Ações da Apple')
plt.xlabel('Ano')
plt.ylabel('Preço Médio Anual ($)')
plt.legend()
plt.grid(True)

def create_pdf(annual_mean, forecast, adjusted_forecast, sentiment, ticker):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, f"Relatório de Previsão e Análise de Sentimento para {ticker}", ln=1, align='C')

    # Adicionar texto
    pdf.ln(10)
    pdf.cell(200, 10, "Previsões:", ln=1)
    current_year = datetime.now().year
    for i, year in enumerate(range(current_year, current_year + 8)):
        pdf.cell(200, 10, f"Previsão sem ajuste para {year}: ${forecast[i]:.2f}", ln=1)
        pdf.cell(200, 10, f"Previsão ajustada para {year}: ${adjusted_forecast[i]:.2f}", ln=1)

    pdf.cell(200, 10, f"Sentimento médio para {ticker}: {sentiment}", ln=1)

    # Adicionar tabela
    pdf.ln(10)
    pdf.cell(200, 10, "Tabela de Previsões:", ln=1)
    data = [["Ano", "Previsão sem ajuste", "Previsão ajustada"]]
    for i, year in enumerate(range(current_year, current_year + 8)):
        data.append([str(year), f"${forecast[i]:.2f}", f"${adjusted_forecast[i]:.2f}"])

    pdf_file_path = "table.pdf"
    pdf_file = SimpleDocTemplate(
        pdf_file_path, pagesize=letter
    )
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements = []
    elements.append(table)
    pdf_file.build(elements)

    # Salvar o PDF
    pdf_file_name = f"{ticker}_Forecast_Sentiment_Report.pdf"
    pdf.output(pdf_file_name)

# Chame a função para criar o PDF
for ticker in tickers:
    annual_mean, forecast, adjusted_forecast, sentiment = predictions[ticker].values()
    create_pdf(annual_mean, forecast, adjusted_forecast, sentiment, ticker)

# Salvar o gráfico em um arquivo PNG
filename = "AAPL_NSRGY_annual_stock_price_ARIMA_sentiment_prediction.png"
plt.savefig(filename, dpi=300)
print(f"\nGráfico salvo como {filename}")

plt.show()