# Séries Temporais

Este repositório reúne os materiais desenvolvidos na disciplina de **Séries Temporais** da minha graduação. Os trabalhos abordam diferentes modelos de previsão temporal, combinando fundamentação teórica e aplicação prática em dados reais.

---

## Trabalho 1 – Modelo Prophet (META)

**Objetivo:** escolher um modelo não apresentado em aula, explicar sua teoria e ilustrá-lo em dados reais.

**Modelo escolhido:** Prophet (Taylor & Letham, 2018), um modelo aditivo generalizado com detecção automática de changepoints, sazonalidade por séries de Fourier e suporte a feriados.

**Aplicação:** índice de produção industrial do Brasil (IBGE/PIM-PF, série 21859 do Banco Central), período de janeiro/2002 a janeiro/2026.

### Conteúdo

- `trabalho_1.py` – Código completo em Python (análise exploratória, ajuste do modelo, métricas, benchmarks, validação cruzada, previsão futura e diagnósticos)
- `trabalho_1.tex` – Relatório teórico em LaTeX (formulação matemática, resultados, discussão crítica e referências)
- `01_serie.png` a `12_previsao.png` – Figuras geradas pelo script

### Como executar

```bash
pip install pandas numpy matplotlib prophet statsmodels scipy scikit-learn
python trabalho_1.py
```

O script baixa os dados diretamente da API do Banco Central e salva todas as figuras.

### Principais resultados

- **MAPE no teste (58 meses):** 3,79%
- **Benchmark Sazonal-Naïve:** 1,76% (sazonalidade domina a série)
- **Resíduos:** normais, mas com autocorrelação e viés sistemático (limitação do modelo)

---

## Trabalho 2 – (em breve)

Em desenvolvimento. O conteúdo será adicionado assim que concluído.

---

**Autor:** Arthur Lourenço Narcizo da Silva
**Disciplina:** Séries Temporais – UNICAMP

---
