import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

df = pd.read_csv("dataset_23.csv")

df_numerico = df.select_dtypes(include='number')
estatisticas = pd.DataFrame({
    'Média': df_numerico.mean(),
    'Mediana': df_numerico.median(),
    'Mínimo': df_numerico.min(),
    'Máximo': df_numerico.max()
})
print("\n=== Estatísticas Descritivas ===")
print(estatisticas)

df_dummies = pd.get_dummies(df, columns=['sistema_operacional', 'tipo_hd', 'tipo_processador'], drop_first=True)

df_dummies.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
df_dummies.dropna(inplace=True)

y = df_dummies['tempo_resposta'].astype(float)
X = df_dummies.drop(columns=['tempo_resposta']).astype(float)
X = sm.add_constant(X)

modelo = sm.OLS(y, X).fit()
print("\n=== Regressão Linear Múltipla ===")
print(modelo.summary())

print("\n=== Diagnóstico de Multicolinearidade (VIF) ===")
vif_data = pd.DataFrame()
vif_data["Variável"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

residuos = modelo.resid
valores_ajustados = modelo.fittedvalues

plt.figure(figsize=(8, 5))
sns.scatterplot(x=valores_ajustados, y=residuos)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Valores Ajustados")
plt.ylabel("Resíduos")
plt.title("Resíduos vs Valores Ajustados")
plt.grid(True)
plt.tight_layout()
plt.show()

bp_test = het_breuschpagan(residuos, X)
labels = ['LM Statistic', 'LM p-value', 'F-statistic', 'F p-value']
bp_resultado = dict(zip(labels, bp_test))

print("\n=== Teste de Breusch-Pagan ===")
for chave, valor in bp_resultado.items():
    print(f"{chave}: {valor:.4f}")
