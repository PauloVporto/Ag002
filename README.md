# Classificador de Espécies de Pinguins

Projeto desenvolvido para a AG2 de Engenharias de Computação e Software.

## O que o projeto faz

- Lê o arquivo CSV com a biblioteca Pandas.
- Remove linhas com valores ausentes.
- Converte os valores categóricos para números inteiros.
- Reordena as colunas na ordem pedida no enunciado.
- Separa o conjunto em 80% para treino e 20% para teste.
- Treina um modelo de classificação Decision Tree.
- Exibe a acurácia e o classification report.
- Permite classificar novos dados pelo terminal ou pela interface web.

## Estrutura

- `app.py`: interface web em Streamlit.
- `main.py`: versão em terminal com `input()`.
- `src/penguin_model.py`: leitura, pré-processamento, treino e previsão.
- `data/penguins.csv`: base de dados utilizada no projeto.

## Como executar

### Interface web

```bash
streamlit run app.py
```

### Versão terminal

```bash
python main.py
```

## Observação

O dataset original utiliza o nome `Adelie`, mas o enunciado traz `Adeline`.
No projeto, os dois nomes são tratados como a mesma espécie para evitar conflito com a atividade.
