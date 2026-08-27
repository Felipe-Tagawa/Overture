# Overture (nome temporário)

Projeto dedicado à primeira etapa da disciplina de Sistemas Operacionais (C12).

## Integrantes

Alunos: Pedro Henrique Ribeiro Dias e Felipe Tagawa Reis
Orientador: Jonas Lopes de Vilas Boas

## Status Atual

Fase de exploração — dataset e abordagem de modelagem ainda em definição.

## Escopo Geral

O projeto tem por objetivo aplicar os conceitos de multithreads para otimização de hiperparâmetros de modelos preditivos, visando identificar os modelos que obtiveram maior acurácia nos testes. Atuando como uma forma de benchmark para encontrar os modelos que melhor satisfizerem as condições impostas.

> Nota: o tipo exato de tarefa de ML (predição, classificação ou clusterização) ainda não foi definido — depende da escolha final do dataset e das colunas disponíveis. A métrica de avaliação (ex: acurácia) será ajustada conforme essa decisão.

Nota: Não visamos alcançar maior desempenho de eficiência durante o treinamento dos modelos, visto que estes estarão rodando em threads independentes, não usufruindo das melhores qualidades de ambiente (hardware) para alcançar máximo desempenho de execução.

## Dataset (em avaliação)

- **Principal candidato:** Asteroid Dataset (Kaggle)
- Em paralelo, está em construção um `schema.toml` para mapeamento e documentação de todas as colunas do dataset, dado o vocabulário técnico específico da área (astrofísica).

## Abordagens em avaliação (multithread)

Observações levantadas pelo professor como quesitos desejáveis a serem aprimorados no escopo do projeto, dado que o foco é a utilização de threads. As alterações e decisão final sobre qual opção seguir ainda não foram tomadas.

1. Utilização de threads para potencializar modelos dado um tempo finito
2. Utilização de threads para treinar modelos diferentes e observar qual foi mais acurado
3. Potencializar treinamento de um único modelo com multiprocessamento das threads para aumentar a quantidade de dados processados simultaneamente

## Tecnologias utilizadas

- Python
- Demais bibliotecas/ferramentas: a definir conforme escolha do dataset e da abordagem de modelagem

## Observações do orientador

Ver seção "Abordagens em avaliação" acima.