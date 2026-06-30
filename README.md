# ecommerce-sentiment-analysis
# Caso
Imagine que você é o gerente de uma empresa de e
commerce e deseja entender melhor as opiniões dos clientes sobre os produtos vendidos. 
Você tem milhares de avaliações de clientes armazenadas em seu banco de dados
e precisa extrair insights sobre a satisfação dos clientes. A
solução envolve a utilização de técnicas de processamento
de linguagem natural (NLP) para analisar as avaliações e
visualizar os resultados de forma clara e informativa. 
Como podemos fazer essa análise?

# Análise de Sentimento em E-commerce com NLP

Este projeto analisa avaliações de clientes de e-commerce utilizando Processamento de Linguagem Natural (NLP) para extrair insights sobre a satisfação dos produtos e os termos mais frequentes.

## 🚀 Evolução do Projeto

O projeto foi desenvolvido em duas etapas para demonstrar boas práticas de refatoração e engenharia de software:

### 1. Versão Inicial (`01_versao_inicial.py`)
* **Abordagem**: Script sequencial focado na validação rápida do pipeline de dados.
* **Limitações Identificadas**: Código rígido (difícil de testar), bloqueio na renderização consecutiva de telas do Matplotlib e risco de inversão lógica na cor dos gráficos.

### 2. Versão Definitiva (`02_versao_final.py`)
* **Melhorias**: 
  * **Modularização**: Código totalmente componentizado em funções de responsabilidade única.
  * **Interface Fluida**: Implementação de `plt.subplots()` para renderizar os gráficos juntos em uma única janela.
  * **Robustez**: Correção do mapeamento de cores por dicionário e limitação do Bag-of-Words para exibir apenas as Top 15 palavras, evitando poluição visual.
