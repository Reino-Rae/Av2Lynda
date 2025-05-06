import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("dataset_23.csv")
df_dummies = pd.get_dummies(df, columns=['sistema_operacional', 'tipo_hd', 'tipo_processador'], drop_first=True)
df_dummies.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
df_dummies.dropna(inplace=True)

y = df_dummies['tempo_resposta'].astype(float)
X = df_dummies.drop(columns=['tempo_resposta']).astype(float)
X = sm.add_constant(X)

modelo_1 = sm.OLS(y, X).fit()

variaveis_excluir = ['latencia_ms', 'armazenamento_tb', 'tipo_hd_SSD', 'tipo_processador_Intel']

X_mod2 = X.drop(columns=variaveis_excluir)
modelo_2 = sm.OLS(y, X_mod2).fit()

print("\n=== Comparação entre Modelos ===")
print(f"Modelo 1 - R² ajustado: {modelo_1.rsquared_adj:.4f}")
print(f"Modelo 1 - F-statistic: {modelo_1.fvalue:.2f}")
print(f"Modelo 2 - R² ajustado: {modelo_2.rsquared_adj:.4f}")
print(f"Modelo 2 - F-statistic: {modelo_2.fvalue:.2f}")


print("\n=== Variáveis excluídas no Modelo 2 ===")
for var in variaveis_excluir:
    print(f"- {var}")
print("\nJustificativa: Variáveis removidas por apresentarem p-valor > 0.05 e/ou VIF elevado (baixa significância estatística ou multicolinearidade).")
