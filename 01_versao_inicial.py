import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer

# 1
# Dados fictícios de avaliações de clientes
data = {
    'text': [
        'I love this product, it`s excellent!',
        'The product is good, but it could be better.',
        'I didn`t like the product, very bad.',
        'Great quality product, I recommend it!',
        'Terrible, it doesn`t work as expected.',
        'It exceeded all my expectetion, worth every penny',
        'I`ve been using it for weeks and the result is amazing.',
        'Very easy to use and the customer support is wanderful.',
        'I will definitely buy again, ten out of ten!',
        'The material is highly durable and of premium quality.',
        'It arrived defective and te box was crushed.',
        'I loved the color e the size fits perfectly.',
        'The product is smaller than it looks in the photos.',
        'The product does what it promises, but doesn`t bring anything new.',
        'It arrived on time and matches the advertisement description.'
    ]
}

df = pd.DataFrame(data)

# 2
# Analisando sentimento
df['sentiment'] = df['text'].apply( lambda x: 
TextBlob(x).sentiment.polarity)

df['sentiment_label'] = df['sentiment'].apply( lambda x: 'positivo' if x > 0
else ('negativo' if x < 0 else 'neutro'))
print(df)

# 3
# Contar a frequência de cada sentimento

sentiment_counts = df['sentiment_label'].value_counts()

# # Criar o gráfico de barras
sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
plt.title('Distribuição dos Sentimentos das Avaliações')
plt.xlabel('Sentimento')
plt.ylabel('Frequência')
plt.show()

# 4
# Inicializar o CountVectorizer
vectorizer = CountVectorizer(stop_words = 'english')

# Ajustar e transformar os dados
X = vectorizer.fit_transform(df['text'])

# Converter para um DataFrame
df_bow = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print(df_bow)

# Visualizar a frequência de palavras
word_freq = df_bow.sum(axis=0)
word_freq.sort_values(ascending=False).plot(kind='bar', figsize=(10, 6))
plt.title('Frequência de Palavras')
plt.xlabel('Palavras')
plt.ylabel('Frequência')
plt.show()

