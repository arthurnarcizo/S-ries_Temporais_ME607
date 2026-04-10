"""
Projeto 1 - Modelo Prophet
Serie: Producao Industrial (IBGE/BCB Serie 21859)

Para rodar:
  pip install pandas numpy matplotlib prophet statsmodels scipy scikit-learn
  python projeto1_prophet.py

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric, add_changepoints_to_plot
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera, probplot, norm, ttest_1samp, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Configuracao dos graficos
plt.rcParams.update({'figure.figsize': (12, 5), 'font.size': 11,
                     'axes.grid': True, 'grid.alpha': 0.3})

# 1. LEITURA DOS DADOS

print("1. Lendo dados do BCB...")

url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.21859/dados?formato=csv"
raw = pd.read_csv(url, sep=';', decimal=',')
raw.columns = ['data', 'valor']
raw['data'] = pd.to_datetime(raw['data'], format='%d/%m/%Y')
raw['valor'] = pd.to_numeric(raw['valor'], errors='coerce')
raw = raw.dropna().sort_values('data').reset_index(drop=True)

# Prophet precisa de colunas ds e y
df = raw.rename(columns={'data': 'ds', 'valor': 'y'})

print(f"   Periodo: {df['ds'].min().strftime('%m/%Y')} a {df['ds'].max().strftime('%m/%Y')}")
print(f"   Observacoes: {len(df)}")
print(f"   Media: {df['y'].mean():.2f}  Desvio: {df['y'].std():.2f}")

# Grafico da serie
fig, ax = plt.subplots()
ax.plot(df['ds'], df['y'], lw=0.9, color='steelblue')
ax.set_title('Producao Industrial - Brasil (IBGE/BCB)')
ax.set_ylabel('Indice (base=100)')
ax.set_xlabel('Data')
plt.tight_layout()
plt.savefig('01_serie.png', dpi=150)
plt.show()

# 2. ANALISE EXPLORATORIA
print("\n2. Analise exploratoria...")

# Criar serie temporal indexada
df_ts = raw.set_index('data').asfreq('MS')
df_ts.columns = ['y']
if df_ts['y'].isna().any():
    df_ts['y'] = df_ts['y'].interpolate()

# Decomposicao STL
stl = STL(df_ts['y'], period=12, robust=True)
res_stl = stl.fit()

fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
componentes = [
    (res_stl.observed, 'Observado', 'steelblue'),
    (res_stl.trend, 'Tendencia', 'red'),
    (res_stl.seasonal, 'Sazonalidade', 'green'),
    (res_stl.resid, 'Residuo', 'gray')
]
for ax, (comp, nome, cor) in zip(axes, componentes):
    ax.plot(comp, color=cor, lw=0.9)
    ax.set_ylabel(nome)
axes[0].set_title('Decomposicao STL')
plt.tight_layout()
plt.savefig('02_stl.png', dpi=150)
plt.show()

# Forca da tendencia e sazonalidade
resid_c = res_stl.resid.dropna()
trend_c = res_stl.trend.reindex(resid_c.index)
seas_c = res_stl.seasonal.reindex(resid_c.index)
F_t = max(0, 1 - np.var(resid_c) / np.var(resid_c + trend_c))
F_s = max(0, 1 - np.var(resid_c) / np.var(resid_c + seas_c))
print(f"   Forca tendencia:    {F_t:.4f}")
print(f"   Forca sazonalidade: {F_s:.4f}")

# Teste ADF
print("   Testes ADF:")
series_adf = [('Nivel', df_ts['y']),
              ('1a Diferenca', df_ts['y'].diff()),
              ('Dif. Sazonal', df_ts['y'].diff(12))]
for nome, s in series_adf:
    resultado = adfuller(s.dropna(), autolag='AIC')
    estac = "Estacionaria" if resultado[1] < 0.05 else "Nao estacionaria"
    print(f"     {nome}: p-valor={resultado[1]:.6f} -> {estac}")

# Boxplot mensal
fig, ax = plt.subplots(figsize=(10, 5))
meses = ['Jan','Fev','Mar','Abr','Mai','Jun',
         'Jul','Ago','Set','Out','Nov','Dez']
dados_mes = [df_ts[df_ts.index.month == m]['y'].values for m in range(1, 13)]
bp = ax.boxplot(dados_mes, labels=meses, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('steelblue')
    patch.set_alpha(0.6)
ax.set_title('Sazonalidade Mensal')
ax.set_ylabel('Indice')
plt.tight_layout()
plt.savefig('03_boxplot.png', dpi=150)
plt.show()

# ACF e PACF
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(df_ts['y'].dropna(), lags=36, ax=axes[0], alpha=0.05)
axes[0].set_title('ACF')
plot_pacf(df_ts['y'].dropna(), lags=36, ax=axes[1], alpha=0.05, method='ywm')
axes[1].set_title('PACF')
plt.tight_layout()
plt.savefig('04_acf_pacf.png', dpi=150)
plt.show()

# 3. AJUSTE DO PROPHET
print("\n3. Ajustando Prophet...")

# Divisao treino/teste (80/20)
corte = int(len(df) * 0.80)
treino = df.iloc[:corte]
teste = df.iloc[corte:]
print(f"   Treino: {len(treino)} obs ({treino['ds'].iloc[0].strftime('%m/%Y')} a {treino['ds'].iloc[-1].strftime('%m/%Y')})")
print(f"   Teste:  {len(teste)} obs ({teste['ds'].iloc[0].strftime('%m/%Y')} a {teste['ds'].iloc[-1].strftime('%m/%Y')})")

# Datas do carnaval (afeta producao industrial)
datas_carnaval = [
    '2002-02-12','2003-03-04','2004-02-24','2005-02-08',
    '2006-02-28','2007-02-20','2008-02-05','2009-02-24',
    '2010-02-16','2011-03-08','2012-02-21','2013-02-12',
    '2014-03-04','2015-02-17','2016-02-09','2017-02-28',
    '2018-02-13','2019-03-05','2020-02-25','2021-02-16',
    '2022-03-01','2023-02-21','2024-02-13','2025-03-04'
]
carnaval = pd.DataFrame({
    'holiday': 'carnaval',
    'ds': pd.to_datetime(datas_carnaval),
    'lower_window': 0,
    'upper_window': 0
})

# Criar e ajustar modelo
modelo = Prophet(
    growth='linear',
    changepoint_prior_scale=0.1,
    seasonality_prior_scale=10.0,
    seasonality_mode='additive',
    yearly_seasonality=8,
    weekly_seasonality=False,
    daily_seasonality=False,
    interval_width=0.95,
    holidays=carnaval,
    changepoint_range=0.85
)
modelo.fit(treino)
print(f"   Changepoints detectados: {len(modelo.changepoints)}")

# Fazer previsao
futuro = modelo.make_future_dataframe(periods=len(teste), freq='MS')
prev = modelo.predict(futuro)

# Grafico dos componentes
fig = modelo.plot_components(prev, figsize=(12, 10))
plt.suptitle('Componentes do Prophet', fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('05_componentes.png', dpi=150)
plt.show()

# Grafico com changepoints
fig = modelo.plot(prev, figsize=(12, 5))
add_changepoints_to_plot(fig.gca(), modelo, prev)
plt.title('Ajuste + Changepoints')
plt.ylabel('Indice')
plt.tight_layout()
plt.savefig('06_changepoints.png', dpi=150)
plt.show()

# 4. METRICAS
print("\n4. Calculando metricas...")

# Juntar previsao com valores reais do teste
pt = prev[prev['ds'].isin(teste['ds'])].merge(teste, on='ds')
y_real = pt['y'].values
y_prev = pt['yhat'].values

# Calcular metricas
rmse = np.sqrt(mean_squared_error(y_real, y_prev))
mae = mean_absolute_error(y_real, y_prev)
mape = mean_absolute_percentage_error(y_real, y_prev) * 100
ss_res = np.sum((y_real - y_prev)**2)
ss_tot = np.sum((y_real - y_real.mean())**2)
r2 = 1 - ss_res / ss_tot
theil = np.sqrt(np.mean((y_real - y_prev)**2)) / (np.sqrt(np.mean(y_real**2)) + np.sqrt(np.mean(y_prev**2)))
cobertura = np.mean((y_real >= pt['yhat_lower'].values) & (y_real <= pt['yhat_upper'].values)) * 100

print(f"   RMSE:         {rmse:.2f}")
print(f"   MAE:          {mae:.2f}")
print(f"   MAPE:         {mape:.2f}%")
print(f"   R2:           {r2:.4f}")
print(f"   U Theil:      {theil:.6f}")
print(f"   Cobertura IC: {cobertura:.1f}%")

# Grafico real vs previsto
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(pt['ds'], y_real, label='Real', lw=1.5, color='steelblue', marker='o', ms=3)
ax.plot(pt['ds'], y_prev, label='Prophet', lw=1.5, color='red', ls='--', marker='s', ms=3)
ax.fill_between(pt['ds'], pt['yhat_lower'], pt['yhat_upper'],
                alpha=0.15, color='red', label='IC 95%')
ax.set_title('Real vs. Prophet (Teste)')
ax.set_ylabel('Indice')
ax.legend()
info = f'RMSE={rmse:.2f}\nMAPE={mape:.2f}%\nR2={r2:.4f}'
ax.text(0.02, 0.97, info, transform=ax.transAxes, fontsize=10,
        va='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.8))
plt.tight_layout()
plt.savefig('07_real_vs_previsto.png', dpi=150)
plt.show()

# Serie completa com ajuste
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df['ds'], df['y'], lw=0.7, color='steelblue', alpha=0.6, label='Observado')
ax.plot(prev['ds'], prev['yhat'], lw=1, color='red', alpha=0.7, label='Prophet')
ax.axvline(treino['ds'].iloc[-1], color='black', ls='--', lw=1, label='Corte treino/teste')
ax.fill_between(prev['ds'], prev['yhat_lower'], prev['yhat_upper'], alpha=0.08, color='red')
ax.set_title('Serie completa + Prophet')
ax.set_ylabel('Indice')
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('08_serie_ajuste.png', dpi=150)
plt.show()

# 5. DIAGNOSTICOS DOS RESIDUOS
print("\n5. Diagnosticos dos residuos...")

residuos = y_real - y_prev

print(f"   Media:      {residuos.mean():.4f}")
print(f"   Mediana:    {np.median(residuos):.4f}")
print(f"   Desvio:     {residuos.std():.4f}")
print(f"   Assimetria: {pd.Series(residuos).skew():.4f}")
print(f"   Curtose:    {pd.Series(residuos).kurtosis():.4f}")

# graficos de diagnostico
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Residuos no tempo
axes[0,0].plot(pt['ds'], residuos, lw=0.9, color='steelblue', marker='o', ms=3)
axes[0,0].axhline(0, color='red', ls='--')
axes[0,0].fill_between(pt['ds'], -2*residuos.std(), 2*residuos.std(),
                       alpha=0.1, color='red', label='+-2s')
axes[0,0].set_title('Residuos no tempo')
axes[0,0].legend(fontsize=8)

# Histograma
axes[0,1].hist(residuos, bins=20, density=True, color='steelblue', alpha=0.7, ec='white')
x_hist = np.linspace(residuos.min(), residuos.max(), 50)
axes[0,1].plot(x_hist, norm.pdf(x_hist, residuos.mean(), residuos.std()),
               'r-', lw=2, label='Normal teorica')
axes[0,1].set_title('Histograma')
axes[0,1].legend(fontsize=8)

# QQ-Plot
probplot(residuos, plot=axes[1,0])
axes[1,0].set_title('QQ-Plot')

# ACF dos residuos
n_lags = min(24, len(residuos)//2 - 1)
if n_lags > 1:
    plot_acf(residuos, lags=n_lags, ax=axes[1,1], alpha=0.05)
axes[1,1].set_title('ACF dos residuos')

plt.suptitle('Diagnosticos dos Residuos', fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('09_diagnosticos.png', dpi=150)
plt.show()

# Testes estatisticos
print("\n   Testes estatisticos:")

# Jarque-Bera
jb_stat, jb_p = jarque_bera(residuos)
jb_result = "Normal" if jb_p > 0.05 else "Nao normal"
print(f"   Jarque-Bera: stat={jb_stat:.4f}, p={jb_p:.6f} -> {jb_result}")

# Teste t
t_stat, t_p = ttest_1samp(residuos, 0)
t_result = "Sem vies" if t_p > 0.05 else "Vies"
print(f"   Teste t:     t={t_stat:.4f}, p={t_p:.6f} -> {t_result}")

# Ljung-Box
lags_testar = [6, 12, 24]
lb = acorr_ljungbox(residuos, lags=lags_testar, return_df=True)
print("   Ljung-Box:")
for lag in lags_testar:
    p_lb = lb.loc[lag, 'lb_pvalue']
    lb_result = "Sem autocorr." if p_lb > 0.05 else "Autocorr."
    print(f"     Lag {lag}: Q={lb.loc[lag,'lb_stat']:.4f}, p={p_lb:.4f} -> {lb_result}")

# Spearman
sp_rho, sp_p = spearmanr(np.arange(len(residuos)), np.abs(residuos))
sp_result = "Homoced." if sp_p > 0.05 else "Heteroced."
print(f"   Spearman:    rho={sp_rho:.4f}, p={sp_p:.4f} -> {sp_result}")

# 6. VALIDACAO CRUZADA
print("\n6. Validacao cruzada...")

modelo_cv = Prophet(
    growth='linear',
    changepoint_prior_scale=0.1,
    seasonality_prior_scale=10.0,
    seasonality_mode='additive',
    yearly_seasonality=8,
    weekly_seasonality=False,
    daily_seasonality=False,
    holidays=carnaval,
    changepoint_range=0.85
)
modelo_cv.fit(df)

print("   Executando (pode demorar um pouco)...")
df_cv = cross_validation(modelo_cv, initial='1460 days',
                         period='365 days', horizon='365 days')
df_perf = performance_metrics(df_cv)

mape_cv = df_perf['mape'].mean() * 100
rmse_cv = df_perf['rmse'].mean()
print(f"   MAPE medio CV: {mape_cv:.2f}%")
print(f"   RMSE medio CV: {rmse_cv:.2f}")

fig = plot_cross_validation_metric(df_cv, metric='mape', figsize=(12, 5))
plt.title('MAPE por Horizonte (Validacao Cruzada)')
plt.tight_layout()
plt.savefig('10_cv_mape.png', dpi=150)
plt.show()

fig = plot_cross_validation_metric(df_cv, metric='rmse', figsize=(12, 5))
plt.title('RMSE por Horizonte (Validacao Cruzada)')
plt.tight_layout()
plt.savefig('10b_cv_rmse.png', dpi=150)
plt.show()

# 7. BENCHMARKS
print("\n7. Comparando com benchmarks...")

resultados = {}

# Prophet
resultados['Prophet'] = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

# Naive: previsao = valor anterior
y_naive = y_real[:-1]
y_naive_real = y_real[1:]
resultados['Naive'] = {
    'RMSE': np.sqrt(mean_squared_error(y_naive_real, y_naive)),
    'MAE': mean_absolute_error(y_naive_real, y_naive),
    'MAPE': mean_absolute_percentage_error(y_naive_real, y_naive) * 100
}

# Sazonal-Naive: previsao = valor de 12 meses atras
if len(y_real) > 12:
    y_sn_real = y_real[12:]
    y_sn_prev = y_real[:-12]
    resultados['S-Naive'] = {
        'RMSE': np.sqrt(mean_squared_error(y_sn_real, y_sn_prev)),
        'MAE': mean_absolute_error(y_sn_real, y_sn_prev),
        'MAPE': mean_absolute_percentage_error(y_sn_real, y_sn_prev) * 100
    }

# Media: previsao = media dos valores
media_val = y_real.mean()
resultados['Media'] = {
    'RMSE': np.sqrt(mean_squared_error(y_real, np.full_like(y_real, media_val))),
    'MAE': mean_absolute_error(y_real, np.full_like(y_real, media_val)),
    'MAPE': mean_absolute_percentage_error(y_real, np.full_like(y_real, float(media_val))) * 100
}

# Drift: reta entre primeiro e ultimo valor do treino
inclinacao = (treino['y'].iloc[-1] - treino['y'].iloc[0]) / (len(treino) - 1)
y_drift = np.array([treino['y'].iloc[-1] + inclinacao*(i+1) for i in range(len(teste))])
resultados['Drift'] = {
    'RMSE': np.sqrt(mean_squared_error(y_real, y_drift)),
    'MAE': mean_absolute_error(y_real, y_drift),
    'MAPE': mean_absolute_percentage_error(y_real, y_drift) * 100
}

# Mostrar resultados
print(f"\n   {'Modelo':<10} {'RMSE':>8} {'MAE':>8} {'MAPE':>8}")
print("   " + "-" * 38)
for nome, vals in resultados.items():
    print(f"   {nome:<10} {vals['RMSE']:>8.2f} {vals['MAE']:>8.2f} {vals['MAPE']:>7.2f}%")

# Grafico de barras
fig, ax = plt.subplots(figsize=(9, 5))
nomes = list(resultados.keys())
mapes = [resultados[n]['MAPE'] for n in nomes]
cores = ['red' if n == 'Prophet' else 'steelblue' for n in nomes]
barras = ax.bar(nomes, mapes, color=cores, alpha=0.8, ec='black')
for barra, val in zip(barras, mapes):
    ax.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.2,
            f'{val:.1f}%', ha='center', fontsize=9)
ax.set_title('MAPE - Prophet vs Benchmarks')
ax.set_ylabel('MAPE (%)')
plt.tight_layout()
plt.savefig('11_benchmarks.png', dpi=150)
plt.show()

# 8. PREVISAO FUTURA (12 MESES)
print("\n8. Previsao 12 meses a frente...")

modelo_final = Prophet(
    growth='linear',
    changepoint_prior_scale=0.1,
    seasonality_prior_scale=10.0,
    seasonality_mode='additive',
    yearly_seasonality=8,
    weekly_seasonality=False,
    daily_seasonality=False,
    interval_width=0.95,
    holidays=carnaval,
    changepoint_range=0.85
)
modelo_final.fit(df)

futuro_12 = modelo_final.make_future_dataframe(periods=12, freq='MS')
prev_12 = modelo_final.predict(futuro_12).tail(12)

print(f"\n   Ultimo dado: {df['ds'].iloc[-1].strftime('%m/%Y')} (indice = {df['y'].iloc[-1]:.2f})")
print(f"\n   {'Mes':<10} {'Previsao':>10} {'IC inf':>10} {'IC sup':>10}")
print("   " + "-" * 43)
for _, linha in prev_12.iterrows():
    print(f"   {linha['ds'].strftime('%m/%Y'):<10} {linha['yhat']:>10.2f} "
          f"{linha['yhat_lower']:>10.2f} {linha['yhat_upper']:>10.2f}")

# Grafico da previsao
fig, ax = plt.subplots(figsize=(14, 5))
ultimos = df.tail(60)
ax.plot(ultimos['ds'], ultimos['y'], lw=1.3, color='steelblue',
        marker='o', ms=3, label='Historico (5 anos)')
ax.plot(prev_12['ds'], prev_12['yhat'], lw=2.5, color='red',
        ls='--', marker='D', ms=5, label='Previsao (12 meses)')
ax.fill_between(prev_12['ds'], prev_12['yhat_lower'], prev_12['yhat_upper'],
                alpha=0.2, color='red', label='IC 95%')
ax.axvline(df['ds'].iloc[-1], color='gray', ls='--', alpha=0.7)
ax.set_title('Previsao 12 meses - Producao Industrial')
ax.set_ylabel('Indice')
ax.legend()
plt.tight_layout()
plt.savefig('12_previsao.png', dpi=150)
plt.show()

# RESUMO
print("\n" + "=" * 58)
print("RESUMO")
print("=" * 58)
print(f"Periodo:    {df['ds'].min().strftime('%m/%Y')} a {df['ds'].max().strftime('%m/%Y')}")
print(f"Obs:        {len(df)} (treino: {len(treino)}, teste: {len(teste)})")
print(f"RMSE:       {rmse:.2f}")
print(f"MAE:        {mae:.2f}")
print(f"MAPE:       {mape:.2f}%")
print(f"R2:         {r2:.4f}")
print(f"U Theil:    {theil:.6f}")
print(f"Cobertura:  {cobertura:.1f}%")
print(f"MAPE CV:    {mape_cv:.2f}%")
print(f"RMSE CV:    {rmse_cv:.2f}")
print("Figuras salvas: 01 a 12")
