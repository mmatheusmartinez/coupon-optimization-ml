# Coupon Optimization - Offer Prioritization System

Case técnico de Data Science desenvolvido com o objetivo de utilizar dados históricos de clientes, transações e campanhas para apoiar duas decisões:

1. quais clientes devem ser priorizados em uma campanha;
2. qual oferta apresenta maior valor esperado entre as opções disponíveis.

A solução combina processamento distribuído em PySpark, modelagem supervisionada com LightGBM e uma simulação retrospectiva de impacto promocional.

---

## Visão Geral da Solução

O projeto foi estruturado em quatro etapas:

1. limpeza e normalização dos dados;
2. reconstrução dos episódios de oferta;
3. criação de features anteriores ao momento do envio;
4. estimação da probabilidade de resposta e priorização dos clientes.

```text
Dados brutos
    ↓
Silver (eventos normalizados)
    ↓
Gold (um envio/episódio de oferta por linha)
    ↓
Feature engineering
    ↓
Regressão Logística e LightGBM
    ↓
Score de propensão e simulação de campanha
```

---

## Dados

O conjunto contém:

* 17.000 clientes;
* 10 ofertas;
* 306.534 eventos;
* 30 dias de acompanhamento;
* eventos de recebimento, visualização, conclusão e transação.

Entre as 10 ofertas, 8 possuem incentivo promocional e 2 são informacionais.

---

## Principais Achados

* A taxa de resposta varia de aproximadamente 43% a 69% entre as ofertas promocionais.
* Cerca de 17% das conclusões aconteceram sem visualização registrada dentro da janela de validade.
* As ofertas com maior desconto não foram necessariamente as de melhor desempenho.
* Clientes que o ticket histórico estava abaixo do valor mínimo exigido apresentaram resposta significativamente menor.
* A compatibilidade entre ticket habitual e valor mínimo da oferta foi uma das características mais relevantes para o modelo.

Esses resultados indicam que o desempenho de uma campanha depende não apenas do desconto, mas também do perfil de consumo, da barreira de ativação e das características da oferta.

---

## Arquitetura de Dados

```text
Bronze
offers.json
profile.json
transactions.json

        ↓

Silver
silver_events
Uma linha por evento normalizado

        ↓

Gold
gold_episodios
Uma linha por envio específico de oferta
```

A construção da Gold foi necessária porque o mesmo cliente pode receber a mesma oferta mais de uma vez.

A unidade de análise utilizada foi:

```text
account_id + offer_id + received_time
```

Cada visualização e conclusão foi associada ao recebimento correspondente, respeitando:

* ordem temporal;
* janela de validade;
* múltiplos recebimentos;
* múltiplas visualizações;
* múltiplas conclusões;
* precisão intradiária.

Também foram removidos da modelagem os episódios que a janela de validade não estava integralmente observada até o final do conjunto de dados. A presentaça desses dados impactaria negativamente as estimativas. 

---

## Target

O target principal foi definido como:

```text
successful_response = 1
```

quando o cliente:

1. recebeu a oferta;
2. visualizou após o recebimento;
3. concluiu após a visualização;
4. concluiu dentro da validade.

Conclusões sem visualização registrada foram mantidas para análise, mas não consideradas respostas positivas no target principal.

Essa escolha busca representar uma sequência compatível com exposição e resposta à campanha, sem afirmar causalidade.

---

## Modelagem

Foram comparados dois modelos:

* Regressão Logística, como baseline;
* LightGBM, como modelo principal.

O LightGBM foi selecionado por sua capacidade de capturar relações não lineares e interações entre características dos clientes, comportamento histórico e atributos das ofertas.

O split foi realizado temporalmente por ondas de envio:

* treino: ondas dos dias 0 e 7;
* validação: onda do dia 14;
* teste: onda do dia 17.

Todas as features comportamentais foram calculadas utilizando apenas informações disponíveis antes do recebimento da oferta.

---

## Resultados

| Métrica               | Regressão Logística |  LightGBM |
| --------------------- | ------------------: | --------: |
| ROC-AUC               |               0.752 | **0.803** |
| PR-AUC                |               0.633 | **0.691** |
| Brier Score           |               0.205 | **0.175** |
| F1 da classe positiva |               0.434 | **0.644** |
| Lift@10%              |                   — | **2.02x** |

Entre os 10% de clientes com maior propensão prevista, a taxa de resposta foi aproximadamente duas vezes superior à seleção aleatória.

### Simulação por threshold

| Threshold | Envios | Redução de envios | Respostas preservadas | Taxa de resposta | Custo esperado |
| --------: | -----: | ----------------: | --------------------: | ---------------: | -------------: |
|      0.30 |  6.325 |               38% |                 88,5% |            55,2% |      R$ 17.662 |
|      0.40 |  5.238 |               49% |                 79,5% |            59,9% |      R$ 15.961 |
|      0.50 |  4.063 |               60% |                 67,1% |            65,1% |      R$ 13.425 |

Na onda de teste, o reward histórico observado foi de R$ 29.398.

Os custos da política proposta são estimativas baseadas nas probabilidades previstas e devem ser interpretados como uma simulação retrospectiva, não como economia comprovada.

---

## Aplicação de Negócio

A principal aplicação do modelo é a priorização de públicos.

Os clientes foram separados em faixas de propensão, permitindo identificar grupos com características distintas:

* clientes de baixa propensão apresentam menor ticket, menor compatibilidade com o valor mínimo e menor histórico de resposta;
* clientes de alta propensão apresentam ticket mais elevado, maior histórico de conclusão e maior valor líquido promocional estimado.

Essa segmentação permite direcionar o orçamento promocional para públicos com maior potencial de resposta e reduzir envios de baixa eficiência.

---

## Limitações

### Personalização da oferta

O modelo apresentou forte concentração das recomendações em uma oferta específica, que também possuía o melhor desempenho médio histórico.

Como observamos apenas a oferta efetivamente recebida por cada cliente, não é possível identificar causalmente qual alternativa teria produzido o melhor resultado para a mesma pessoa.

Por isso, a solução é mais robusta para priorizar clientes do que para comprovar personalização individual da oferta.

### Canais

Cada oferta possui uma combinação fixa de canais. Dessa forma, o efeito dos canais não pode ser isolado do efeito da própria oferta.

### Conversões sem visualização

Conclusões sem visualização registrada podem representar compras pouco influenciadas pelo incentivo, mas essa hipótese não pode ser confirmada sem grupo de controle.

---

## Como Reproduzir

### Ambiente

* Databricks Serverless;
* Python 3.12;
* dependências disponíveis em `requirements.txt`.

### Dados

Faça upload dos arquivos para:

```text
/Workspace/Users/<seu_email>/coupon-optimization-ml/data/raw/
```

### Execução

1. Execute `1_data_processing.ipynb`.
2. O notebook criará as tabelas Delta `silver_events` e `gold_episodios`.
3. Execute `2_modeling.ipynb`.
4. O segundo notebook carregará a Gold, construirá as features e executará o pipeline de modelagem.

---

## Estrutura do Repositório

```text
coupon-optimization-ml/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 1_data_processing.ipynb
│   └── 2_modeling.ipynb
├── presentation/
├── requirements.txt
└── README.md
```

---

## Próximos Passos

* executar teste A/B prospectivo para validar o threshold;
* randomizar ofertas entre perfis comparáveis;
* incluir grupo de controle sem incentivo;
* estimar efeito incremental com uplift modeling ou Causal ML;
* explorar contextual bandits para otimização contínua;
* monitorar drift e desempenho a cada nova onda de campanha.
