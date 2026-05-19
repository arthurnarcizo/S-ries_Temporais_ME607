# Séries Temporais

Este repositório reúne os materiais desenvolvidos na disciplina de Séries Temporais do curso de Estatística da UNICAMP (1º Semestre de 2026). Os trabalhos abordam diferentes modelos de previsão e análise temporal, combinando fundamentação teórica e aplicação prática em dados reais do cenário brasileiro.

---

## Trabalho 1 – Modelo Prophet (META)

* **Objetivo:** Escolher um modelo não apresentado formalmente em aula, explicar sua teoria e ilustrá-lo utilizando dados reais.
* **Modelo Escolhido:** Prophet, um modelo aditivo generalizado estruturado com detecção automática de *changepoints*, modelagem de sazonalidade por séries de Fourier e suporte customizado a feriados.
* **Aplicação:** Índice de produção industrial do Brasil (IBGE/PIM-PF, série 21.859 do Banco Central), cobrindo o período de janeiro de 2002 a janeiro de 2026.

### Conteúdo do Trabalho 1
* `trabalho_1.py`: Código completo em Python contendo análise exploratória, ajuste do modelo, métricas de avaliação, benchmarks (*Sazonal-Naïve*), validação cruzada temporal, previsão futura e diagnósticos.
* `trabalho_1.pdf`: Relatório em desenvolvido em LaTeX com a formulação matemática, resultados empíricos, discussões críticas e referências.

### Principais Resultados do Trabalho 1
* **MAPE no conjunto de teste (58 meses):** 3,79%.
* **Benchmark Sazonal-Naïve:** 1,76%, evidenciando que a forte componente sazonal domina a série de produção industrial.
* **Análise de Resíduos:** Resíduos apresentaram comportamento normal, porém com autocorrelação remanescente e viés sistemático, destacando limitações estruturais do Prophet para dependências de curto prazo.

---

## Trabalho 2 – Modelagem da Volatilidade e Análise de Intervenção (PETR4)

* **Objetivo:** Construir um modelo estatístico para modelar a média e a volatilidade condicional de um ativo financeiro de alta liquidez, mensurar o impacto quantitativo de choques político-econômicos via análise de intervenção e validar previsões de risco de mercado.
* **Modelo Escolhido:** Modelagem híbrida baseada na família **ARMA-GARCH** com extensões assimétricas (**EGARCH**) e distribuições de caudas pesadas (**t-Student padronizada**).
* **Aplicação:** Log-retornos diários das ações preferenciais da Petrobras (PETR4), cobrindo o período de 04/01/2010 a 29/04/2026 (total de 4.053 observações).

### Metodologia
A análise exploratória revelou os fatos estilizados clássicos de séries financeiras: forte leptocurtose (curtose em excesso de 13,35), assimetria negativa (-0,93) e alta persistência na autocorrelação dos retornos ao quadrado (efeito ARCH, rejeitado pelo teste ARCH-LM).

Foram comparados três modelos por máxima verossimilhança com inovações $t$-Student:
1. **$GARCH(1,1)-t$:** Modelo simétrico de referência.
2. **$EGARCH(1,1)-t$:** Extensão para capturar o efeito alavancagem (*leverage effect*).
3. **$ARX(1)-EGARCH(1,1)-t$ com Intervenção:** Inclusão de variáveis dummys (pulso e degrau) na equação da média para mensurar o impacto dos seguintes eventos:
   * *Demissão de Roberto Castello Branco (19/02/2021):* Modelado como dummy de **Pulso**.
   * *Mudança na política de preços de combustíveis (08/03/2023):* Modelado como dummy de **Degrau**.
   * *Demissão de Jean Paul Prates (14/05/2024):* Modelado como dummy de **Pulso**.

### Principais Resultados do Trabalho 2
* **Seleção do Modelo:** O modelo com regressores de intervenção foi o escolhido com base no menor Critério de Informação de Akaike (AIC).
* **Efeito Alavancagem:** Confirmado pelo parâmetro $\gamma = -0,0301$, provando que choques negativos (más notícias) aumentam a volatilidade futura mais do que choques positivos de mesma magnitude.
* **Impacto das Intervenções:**
   * A demissão de Roberto Castello Branco em 2021 gerou um impacto imediato de **-6,50 pontos percentuais** no log-retorno diário.
   * A demissão de Jean Paul Prates em 2024 gerou uma queda imediata de **-2,06 pontos percentuais**.
   * A mudança regulatória da política de preços em 2023 **não apresentou efeito estatisticamente significativo**, sugerindo antecipação e eficiência do mercado na forma semiforte.
* **Value-at-Risk (VaR) & Backtesting:** O VaR paramétrico diário foi calculado para os níveis de 95% e 99%. No backtesting, o teste de cobertura incondicional de Kupiec não rejeitou a adequação do modelo para o nível de confiança de 99%, validando sua aplicação prática sob as diretrizes de risco de mercado.

### Conteúdo do Trabalho 2
* `trabalho_2.py`: Roteiro em Python englobando o pipeline estatístico completo: download/limpeza via Yahoo Finance, testes descritivos (Jarque-Bera, ADF, KPSS, ARCH-LM), estimação das equações de média/volatilidade, extração de resíduos padronizados e cálculo/backtesting do VaR.
* `trabalho_2.pdf`: Relatório em LaTeX estruturado com contextualização de mercado, fundamentação de volatilidade condicional, tabelas estatísticas e discussões críticas sobre as limitações do modelo (como a persistência próxima da não-estacionariedade com $\beta = 0,989$).


