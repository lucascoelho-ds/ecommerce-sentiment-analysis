import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob

def carregar_dados():

    data = {
        'text': [
            'I love this product, it`s excellent!',
            'The product is good, but it could be better.',
            'I didn`t like the product, very bad.',
            'Great quality product, I recommend it!',
            'Terrible, it doesn`t work as expected.',
            'It exceeded all my expectation, worth every penny',
            'I`ve been using it for weeks and the result is amazing.',
            'Very easy to use and the customer support is wonderful.',
            'I will definitely buy again, ten out of ten!',
            'The material is highly durable and of premium quality.',
            'It arrived defective and the box was crushed.',
            'I loved the color e the size fits perfectly.',
            'The product is smaller than it looks in the photos.',
            'The product does what it promises, but doesn`t bring anything new.',
            'It arrived on time and matches the advertisement description.'
        ]
    }
    return pd.DataFrame(data)

def analisar_sentimentos(df):

    df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    df['sentiment_label'] = df['sentiment'].apply(
        lambda x: 'positivo' if x > 0 else ('negativo' if x < 0 else 'neutro')
    )
    return df


def extrair_bag_of_words(df):

    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['text'])
    df_bow = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    return df_bow


def plotar_resultados(df, df_bow):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sentiment_counts = df['sentiment_label'].value_counts()
    
    cores_map = {'positivo': '#2ec4b6', 'negativo': '#e71d36', 'neutro': '#ff9f1c'}
    cores_filtradas = [cores_map[sent] for sent in sentiment_counts.index]

    sentiment_counts.plot(kind='bar', color=cores_filtradas, ax=ax1)
    ax1.set_title('Distribuição dos Sentimentos das Avaliações', fontsize=14, pad=15)
    ax1.set_xlabel('Sentimento', fontsize=12)
    ax1.set_ylabel('Frequência', fontsize=12)
    ax1.tick_params(axis='x', rotation=0)

    # GRÁFICO 2: Frequência de Palavras
    word_freq = df_bow.sum(axis=0)
    word_freq.sort_values(ascending=False).head(15).plot(kind='bar', color='#011627', ax=ax2)
    
    ax2.set_title('Top 15 Palavras Mais Frequentes (BoW)', fontsize=14, pad=15)
    ax2.set_xlabel('Palavras', fontsize=12)
    ax2.set_ylabel('Frequência', fontsize=12)
    ax2.tick_params(axis='x', rotation=45) 
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("Iniciando a pipeline de NLP...\n")
    
    # Passos ordenados
    df_avaliacoes = carregar_dados()
    df_avaliacoes = analisar_sentimentos(df_avaliacoes)
    df_palavras = extrair_bag_of_words(df_avaliacoes)
    
    # Print dos resultados no terminal
    print("--- DataFrame com Sentimentos ---")
    print(df_avaliacoes[['text', 'sentiment_label']].head(10))
    print("\n--- Amostra da Matriz Bag of Words ---")
    print(df_palavras.iloc[:6, :6])
    
    print("\nRenderizando gráficos combinados...")
    plotar_resultados(df_avaliacoes, df_palavras)
