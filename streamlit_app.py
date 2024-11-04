import os
import time
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import PyPDF2
import streamlit as st
import plotly.graph_objects as go 
from dotenv import load_dotenv
from botocore.exceptions import ClientError
import boto3
import json
import requests
from textblob import TextBlob
from matplotlib.patches import Wedge
import numpy as np

# Charger les variables d'environnement
load_dotenv('.env')

MODEL_NAME = 'anthropic.claude-3-sonnet-20240229-v1:0'
AWS_REGION = 'us-west-2'

# Configuration des identifiants AWS
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_SESSION_TOKEN'] = os.getenv('AWS_SESSION_TOKEN')
os.environ['REGION_NAME'] = AWS_REGION

news_api_key = "ddd73f9423ad41c2aff5090e8493c142"

# Créer un client pour le service Bedrock
bedrock = boto3.client('bedrock-runtime', region_name=AWS_REGION)

def get_feeling(symbol):
    url = f"https://newsapi.org/v2/everything?q={symbol}&language=en&apiKey={news_api_key}"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    sentiments = [TextBlob(article['description']).sentiment.polarity for article in articles if article['description']]
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    return avg_sentiment * 100

# Fonction pour déterminer la couleur en fonction du score
def get_color(score):
    if score >= 7:
        return 'green'
    elif score >= 4:
        return 'yellow'
    else:
        return 'red'

def get_esg(symbol):
    ticker = yf.Ticker(symbol)
    esg = ticker.sustainability
    data = {
        'Environment': None,
        'Social': None,
        'Governance': None
    }
    if esg is not None:
        data['Environment'] = esg.loc['environmentScore'].values[0] if 'environmentScore' in esg.index else None
        data['Social'] = esg.loc['socialScore'].values[0] if 'socialScore' in esg.index else None
        data['Governance'] = esg.loc['governanceScore'].values[0] if 'governanceScore' in esg.index else None
    return data





# Calcul des pourcentages pour le diagramme circulaire
def sentiment_percentages(symbol):
    avg_sentiment = get_feeling(symbol)
    # On répartit les pourcentages en fonction de l'indice de sentiment
    positive = max(0, avg_sentiment) * 100  # Sentiment positif
    negative = max(0, -avg_sentiment) * 100  # Sentiment négatif
    neutral = 100 - positive - negative      # Sentiment neutre
    return positive, neutral, negative

def display_pie_chart(label,value, max_value):

    if value < max_value:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': label},
            gauge={
                'axis': {'range': [0, max_value]},
                'steps': [
                {'range': [0, max_value * 0.333], 'color': '#F44336'},  # Rouge pour faible
                {'range': [max_value * 0.333, max_value * 0.666], 'color': '#FFEB3B'},  # Jaune pour moyen
                {'range': [max_value * 0.666, max_value], 'color': '#4CAF50'}  # Vert pour élevé
                ],
                'bar': {'color': "black"},
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': value
                }
            }
        ))

        # Afficher le graphique dans Streamlit
        st.plotly_chart(fig)

def get_completion(prompt, retries=3, wait_time=5):
    """Utilise le modèle pour générer une réponse à un prompt donné."""
    inference_config = {
        "temperature": 0.0,
        "maxTokens": 4096  # Limite de tokens augmentée à 4096
    }
    converse_api_params = {
        "modelId": MODEL_NAME,
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": inference_config
    }
    
    for attempt in range(retries):
        try:
            response = bedrock.converse(**converse_api_params)
            if "output" in response and "message" in response['output'] and "content" in response['output']['message']:
                return response['output']['message']['content'][0]['text']
            else:
                return "Erreur: la réponse n'a pas pu être traitée."
        except ClientError as err:
            message = err.response['Error']['Message']
            if "Too many requests" in message:
                st.warning(f"Trop de requêtes. Nouvelle tentative dans {wait_time} secondes.")
                time.sleep(wait_time)
            else:
                st.error(f"Erreur client AWS: {message}")
                break
    return "Erreur: trop de tentatives infructueuses."

# Fonction pour récupérer les données financières
def fetch_financial_data(ticker):
    stock = yf.Ticker(ticker)
    # Récupérer les états financiers
    income_stmt = stock.financials.T
    return income_stmt

# Fonction pour détecter les changements majeurs
def detect_significant_changes(dataframe, threshold=0.2):
    changes = dataframe.pct_change().dropna()
    significant_changes = changes[changes.abs() > threshold]
    return significant_changes

# Fonction pour récupérer les nouvelles de la compagnie
def fetch_company_news(ticker):
    news = yf.Ticker(ticker).news
    return news

# Interface utilisateur Streamlit
st.set_page_config(layout="wide")  # Utiliser la largeur complète de la page
st.title("Outils d'Analyse Financière")
st.write("Sélectionnez un outil ci-dessous:")


tool = st.selectbox("Choisissez un outil:", ["Surveillance des Indicateurs Financiers", "Analyse de Rapport Annuel", "Comparaison"])




if tool == "Surveillance des Indicateurs Financiers":
    st.subheader("Surveillance des Indicateurs Financiers")
    ticker_input = st.text_input("Symbole boursier:", "AAPL")  # Par défaut, Apple Inc.

    if ticker_input:
        try:
            income_stmt = fetch_financial_data(ticker_input)

            # Visualiser les indicateurs financiers avec des valeurs par défaut
            st.subheader("Dashboard des Indicateurs Financiers")

            # Filtrer les indicateurs financiers disponibles
            indicators = income_stmt.columns.tolist()
            default_indicators = ['Total Revenue', 'Net Income', 'Total Expenses']
            selected_indicators = st.multiselect("Choisissez les indicateurs financiers à afficher:", indicators, default=default_indicators)

            if selected_indicators:
                # Visualiser les indicateurs sélectionnés dans un graphique interactif
                fig = go.Figure()

                for indicator in selected_indicators:
                    fig.add_trace(go.Scatter(
                        x=income_stmt.index,
                        y=income_stmt[indicator],
                        mode='lines+markers',
                        name=indicator
                    ))

                fig.update_layout(title=f'Indicateurs Financiers de {ticker_input}',
                                  xaxis_title='Année',
                                  yaxis_title='Montant',
                                  hovermode='x unified')

                st.plotly_chart(fig)

            # Récupérer les nouvelles de la compagnie
            company_news = fetch_company_news(ticker_input)
            latest_news = company_news[0] if company_news else None
            
            news_title = latest_news['title'] if latest_news else "Aucune nouvelle disponible."
            news_link = latest_news['link'] if latest_news else ""

            # Résumé de la situation financière
            financial_summary_prompt = (
                f"Description de la situation financière récente de {ticker_input} :\n\n"
                f"Données financières : {income_stmt.to_string()}\n\n"
                f"Dernières nouvelles : {news_title}. [Lire plus]({news_link})\n\n"
                "Veuillez que les informations soient concises et faciles à lire. "
                "Ne changez pas la typographie du texte et ne mettez rien en gras."
            )
            financial_summary = get_completion(financial_summary_prompt)
            st.subheader("Description de la situation financière")
            st.write(financial_summary)

            # Analyse des sentiments
            sentiment_prompt = (
                f"Analyse de sentiment des résultats financiers récents de {ticker_input} :\n\n"
                f"Données financières : {income_stmt.to_string()}\n\n"
                f"Dernières nouvelles : {news_title}. [Lire plus]({news_link})\n\n"
                "Veuillez fournir une liste avec 🟢 pour des sentiments positifs, 🟡 pour des sentiments neutres et 🔴 pour des sentiments négatifs. "
                "Les informations doivent être concises et faciles à lire. "
                "Ne changez pas la typographie du texte et ne mettez rien en gras."
            )
            sentiment_analysis = get_completion(sentiment_prompt)
            st.subheader("Analyse de sentiment")

            feeling = get_feeling(ticker_input)
            if feeling > 0:
                display_pie_chart("Sentiment moyen",100 - get_feeling(ticker_input), 100)
            else:
                st.warning("Le sentiment moyen n'est pas disponible pour cette entreprise.")
                
            
            # Afficher l'analyse de sentiment sous forme de liste
            sentiment_lines = sentiment_analysis.split("\n")
            for line in sentiment_lines:
                st.write(line.strip())

            ''
            ''
            
            #ESG
            esg_data = get_esg(ticker_input)
            # Vérification si les scores ESG existent
            st.subheader("Scores ESG")

            if all(value is None for value in esg_data.values()):
                st.warning("Les scores ESG ne sont pas disponibles pour cette entreprise.")
            else:
                # Créer des colonnes pour afficher les jauges horizontalement
                cols = st.columns(len(esg_data))
                
                # Afficher chaque score dans une jauge circulaire
                for col, (label, score) in zip(cols, esg_data.items()):
                    if score is not None:
                        with col:
                            display_pie_chart(label, score, 10)

                esg_prompt = (
                    f"Description du score ESG recent de {ticker_input} :\n\n"
                    f"Données financières : {income_stmt.to_string()}\n\n"
                    f"Données environnementales : {esg_data['Environment']}\n\n"
                    f"Données Sociales : {esg_data['Social']}\n\n"
                    f"Données de gouvernance : {esg_data['Governance']}\n\n"
                    f"Dernières nouvelles : {news_title}. [Lire plus]({news_link})\n\n"
                    "Donne un avis concis sur les scores du esg en tenant compte de la situation financiere de la companie."
                    "Le format doit etre presenté sous forme de points positifs et negatifs"
                    "Veuillez que les informations soient concises et faciles à lire. "
                    "Ne changez pas la typographie du texte et ne mettez rien en gras."
                )
                esg_prompt = get_completion(esg_prompt)
                st.write(esg_prompt)


            
            

            # Projection des prochaines années
            st.subheader("Projection des résultats financiers")
            years_to_project = st.slider("Nombre d'années à projeter:", 1, 5, 1)
            current_year = pd.to_datetime("today").year
            end_year = current_year + years_to_project

            projection_prompt = (
                f"Projection des résultats financiers de {ticker_input} de {current_year} à {end_year}. "
                "Fiez-vous uniquement aux données fournies sans les mentionner, et discutez des perspectives pour cette période. "
                "Faites une analyse concise et facile à lire, point par point, en utilisant des données que vous calculerez par vous-même. "
                "N'écrivez pas comme si vous parliez à un humain, mais plutôt comme si vous écriviez un texte.\n\n"
                "Les informations doivent être concises et faciles à lire. "
                "Ne changez pas la typographie du texte et ne mettez rien en gras."
            )
            projection = get_completion(projection_prompt)
            st.write(projection)

        except Exception as e:
            st.error(f"Une erreur s'est produite lors de la récupération des données: {str(e)}")

elif tool == "Analyse de Rapport Annuel":
    st.subheader("Analyse de Rapport Annuel")
    st.write("Téléchargez un fichier PDF pour obtenir un résumé et une analyse de sentiment.")

    uploaded_file = st.file_uploader("Téléchargez un fichier PDF", type="pdf")

    if uploaded_file is not None:
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # Résumé du contenu
            summary_prompt = (
                f"Résumé des informations clés du rapport:\n\n{text[:2000]}\n\n"
                "Veuillez que les informations soient concises et faciles à lire. "
                "Ne changez pas la typographie du texte et ne mettez rien en gras."
            )
            summary = get_completion(summary_prompt)
            st.subheader("Résumé des informations clés")
            st.write(summary if summary else "Aucun résumé disponible.")
            
            # Analyse de sentiment########################################
            sentiment_prompt = (
                f"Analyse de sentiment du rapport:\n\n{text[:2000]}\n\n"
                "Veuillez que les informations soient concises et faciles à lire. "
                "Ne changez pas la typographie du texte et ne mettez rien en gras."
            )
            sentiment_analysis = get_completion(sentiment_prompt)
            st.subheader("Analyse de sentiment")
            st.write(sentiment_analysis if sentiment_analysis else "Aucune analyse de sentiment disponible.")

        except PyPDF2.errors.DependencyError:
            st.error("Erreur: PyCryptodome est requis pour lire les fichiers PDF sécurisés.")
        except Exception as e:
            st.error(f"Une erreur s'est produite lors de la lecture du fichier PDF: {str(e)}")

elif tool == "Comparaison":
    st.subheader("Comparaison")
    st.write("Sélectionnez les secteurs à comparer et les entreprises.")

    
    companies = st.text_input("Entrez les symboles boursiers des entreprises (séparés par des virgules):", "AAPL, MSFT, GOOGL")

    if st.button("Générer le rapport"):
        if companies:
            companies_list = [ticker.strip() for ticker in companies.split(',')]
            comparison_data = {}
            news_data = {}

            for ticker in companies_list:
                try:
                    income_stmt = fetch_financial_data(ticker)
                    comparison_data[ticker] = {
                        'Total Revenue': income_stmt['Total Revenue'],
                        'Net Income': income_stmt['Net Income'],
                        'Total Expenses': income_stmt['Total Expenses']
                    }
                    news_data[ticker] = fetch_company_news(ticker)

                except Exception as e:
                    st.error(f"Erreur lors de la récupération des données pour {ticker}: {str(e)}")
                    continue
            
            comparison_df = pd.DataFrame(comparison_data)
            st.subheader("Données de Comparaison")
            st.write(comparison_df)

            # Créer un graphique pour les comparaisons
            fig = go.Figure()

            for ticker in companies_list:
                fig.add_trace(go.Bar(
                    x=comparison_df.index,
                    y=comparison_df[ticker],
                    name=ticker
                ))

            fig.update_layout(
                title="Comparaison des Performances Financières",
                xaxis_title="Indicateurs Financiers",
                yaxis_title="Montant",
                barmode='group'
            )
            st.plotly_chart(fig)

            # Générer le rapport de comparaison
            report_prompt = (
                f"Rapport de comparaison pour les entreprises {', '.join(companies_list)} :\n\n"
                f"Données de comparaison :\n{comparison_df.to_string()}\n\n"
                "Comparez les différentes entreprises présentées de manière concise."
                "Veillez à suivre un format sous forme de points positifs et négatifs par rapport aux autres entreprises."
                "Il serait judicieux d'indiquer dans chaque section l'entreprise qui performe le mieux et celle qui performe le moins."
                "Veuillez que les informations soient concises et faciles à lire. "
                "Vers la fin, élaborer davantage sur la performance de chaque en mentionnant comment ils sont meilleurs ou pires que les autres."
                "Ne changez pas la typographie du texte et ne mettez rien en gras."
            )
            report = get_completion(report_prompt)
            st.subheader("Rapport de Comparaison")
            st.write(report)

            # Afficher les nouvelles pour chaque entreprise
            st.subheader("Dernières nouvelles")
            for ticker in companies_list:
                st.write(f"**{ticker}**")
                if ticker in news_data:
                    for article in news_data[ticker]:
                        st.write(f"- {article['title']}: [Lire plus]({article['link']})")
                else:
                    st.write("Aucune nouvelle disponible.")

        else:
            st.warning("Veuillez entrer des symboles boursiers.")