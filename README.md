# Séries Temporais 

Este repositório foi criado para compartilhar os materiais desenvolvidos na disciplina de **Séries Temporais** da minha graduação. O foco principal está na aplicação de modelos de previsão temporal, com ênfase em um modelo não abordado em sala de aula.

## Trabalho 1 – Modelo Prophet (META)

### Objetivo
Escolher um modelo de séries temporais **não apresentado em aula**, explicar seu funcionamento teórico e ilustrar sua aplicação em dados reais.

### Modelo escolhido
**Prophet**, desenvolvido pela equipe do Facebook (atual META), amplamente utilizado para previsões com sazonalidade, feriados e mudanças de tendência.

### Aplicação prática
- **Dados utilizados**: produção industrial do Brasil nos últimos anos.
- **Ferramentas**:
  - Python (biblioteca `prophet`, `pandas`, `numpy`, `matplotlib`)
  - Documentação em LaTeX (relatório completo do trabalho)
- **Etapas realizadas**:
  1. Explicação conceitual do modelo Prophet.
  2. Pré-processamento dos dados de produção industrial.
  3. Ajuste do modelo aos dados históricos.
  4. Geração de previsões futuras.
  5. Avaliação visual e comparativa dos resultados.

### Conteúdo do repositório
- `/notebooks` – Jupyter notebooks com a implementação em Python.
- `/data` – Base de dados da produção industrial (fonte: IBGE/SIDRA).
- `/latex` – Código fonte do relatório em LaTeX e PDF final.
- `/figuras` – Gráficos gerados pelo modelo Prophet.
- `README.md` – Este arquivo.

### Referências utilizadas
- Taylor, S. J., & Letham, B. (2018). *Forecasting at scale*. The American Statistician.
- Materiais apresentados em aula sobre modelos clássicos (ARIMA, ETS).
- Documentação oficial do Prophet: [https://facebook.github.io/prophet/](https://facebook.github.io/prophet/)

### Como reproduzir
1. Clone o repositório.
2. Instale as dependências: `pip install prophet pandas matplotlib`.
3. Execute o notebook principal.
4. Compile o arquivo `.tex` (opcional, para gerar o relatório).
