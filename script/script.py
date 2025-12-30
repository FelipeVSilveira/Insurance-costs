# %% Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# %% Configuração de estilo para os gráficos
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
import warnings
warnings.filterwarnings('ignore')

# %% Carregando o conjunto de dados
df = pd.read_csv('/Users/felipesilveira/VSC/Insurance/data/insurance.csv')

# %% EDA - Análise Exploratória de Dados
# Primeiras linhas
print(df.head())

# Informações gerais
print(df.info())

# Estatísticas descritivas
print(df.describe())

# Verificando valores nulos
print(df.isnull().sum())

# %% Distribuição da variável alvo 'charges'
plt.figure(figsize=(10, 6))
sns.histplot(df['charges'], kde=True, color="#69b3a2", stat="density", linewidth=0)
sns.kdeplot(df['charges'], color="#e63946", linewidth=2)
plt.axvline(df['charges'].mean(), color='red', linestyle='--', label='Média')
plt.axvline(df['charges'].median(), color='blue', linestyle='-', label='Mediana')
plt.title('Distribuição de Custos Médicos (Charges)')
plt.legend()
plt.savefig(os.path.join('images', 'distribuicao_charges.png'), dpi=300, bbox_inches='tight')
plt.show()

# %% Métricas de assimetria e curtose
print(f"Assimetria (Skewness): {stats.skew(df['charges']):.4f}")
print(f"Curtose (Kurtosis): {stats.kurtosis(df['charges']):.4f}")

# %% Boxplot: impacto do tabagismo nos custos médicos
plt.figure(figsize=(8, 6))
sns.boxplot(x='smoker', y='charges', data=df, palette="Set2")
plt.title('Impacto do tabagismo nos custos médicos')
plt.savefig(os.path.join('images', 'boxplot_smoker_charges.png'), dpi=300, bbox_inches='tight')
plt.show()

# %% Scatterplot: Interação BMI e Fumante
plt.figure(figsize=(10, 6))
sns.lmplot(x='bmi', y='charges', hue='smoker', data=df, palette='Set1', height=6, aspect=1.5, scatter_kws={'alpha':0.6})
plt.title('Relação BMI vs Custos (Separado por Fumante)')
plt.savefig(os.path.join('images', 'scatter_bmi_charges.png'), dpi=300, bbox_inches='tight')
plt.show()

# %% Matriz de correlação
df_corr = df.copy()
df_corr['smoker_code'] = df_corr['smoker'].apply(lambda x: 1 if x == 'yes' else 0)
df_corr['sex_code'] = df_corr['sex'].apply(lambda x: 1 if x == 'male' else 0)
corr = df_corr.select_dtypes(include=[np.number]).corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlação')
plt.savefig(os.path.join('images', 'matriz_correlacao.png'), dpi=300, bbox_inches='tight')
plt.show()

# %% Distribuição demográfica(Idade)
plt.figure(figsize=(10, 5))
sns.histplot(df['age'], bins=range(18, 65), kde=True, color="#1650CF", alpha=0.8)
plt.axvspan(17.5, 19.5, color='red', alpha=0.2, label='Pico jovens (18-19 anos)')
plt.title('Distribuição demográfica por idade')
plt.legend()
plt.savefig(os.path.join('images', 'distribuicao_idade.png'), dpi=300, bbox_inches='tight')
plt.show()

# %% Comparativo gênero (Idade e BMI)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.kdeplot(data=df, x='age', hue='sex', fill=True, common_norm=False, palette='coolwarm', ax=axes[0])
axes[0].set_title('Distribuição de Idade por Gênero')

sns.kdeplot(data=df, x='bmi', hue='sex', fill=True, common_norm=False, palette='coolwarm', ax=axes[1])
axes[1].set_title('Distribuição de BMI por Gênero')
plt.tight_layout()
plt.savefig(os.path.join('images', 'distribuicao_genero.png'), dpi=300, bbox_inches='tight')
plt.show()

# %% Interação fumo + obesidade nos custos médicos
df['bmi_category'] = df['bmi'].apply(lambda x: 'Obeso (>=30)' if x >= 30 else 'Não Obeso (<30)')

plt.figure(figsize=(10, 6))
sns.barplot(x='bmi_category', y='charges', hue='smoker', data=df, palette=['#2ecc71', '#e74c3c'], ci=68, capsize=0.1)
plt.title('Quem gasta mais? Interação entre Fumo e Obesidade')
plt.ylabel('Custo Médio ($)')
plt.savefig(os.path.join('images', 'interacao_fumo_obesidade.png'), dpi=300, bbox_inches='tight')
plt.show()

# %% Interação filhos e fumo nos custos médicos
df['smoker_num'] = df['smoker'].apply(lambda x: 1 if x == 'yes' else 0)
smoker_by_children = df.groupby('children')['smoker_num'].mean().reset_index()

plt.figure(figsize=(8, 5))
colors = ['#e74c3c' if x > 0.2 else '#2ecc71' for x in smoker_by_children['smoker_num']]
sns.barplot(x='children', y='smoker_num', data=smoker_by_children, palette=colors)
plt.axhline(0.2, color='black', linestyle='--')
plt.title('Proporção de Fumantes por Quantidade de Filhos')
plt.ylabel('Proporção de Fumantes')
plt.savefig(os.path.join('images', 'proporcao_fumantes_filhos.png'), dpi=300, bbox_inches='tight')
plt.show()

# %% Teste estatístico (T-Test) entre fumantes e não fumantes
smoker_yes = df[df['smoker'] == 'yes']['charges']
smoker_no = df[df['smoker'] == 'no']['charges']
t_stat, p_val = stats.ttest_ind(smoker_yes, smoker_no)
print(f"\nTeste T (Fumantes vs Não Fumantes): P-value = {p_val:.3e}")

# %% Preparação dos dados para modelagem
# %% Separando features e target
X = df.drop(columns=['charges', 'bmi_category', 'smoker_num'])
y = df['charges']

# %% Identificando colunas numéricas e categóricas
num_features = ['age', 'bmi', 'children']
cat_features = ['sex', 'smoker', 'region']

# %% Pipeline de pré-processamento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first'), cat_features)
    ])

# %% Divisão treino/teste( 80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% Modelagem e Avaliação

# %% Modelo de Regressão Linear
model_lr = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', LinearRegression())])
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

# %% Modelo de Random Forest
model_rf = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

# %% Função para calcular métricas de avaliação
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} - MAE: {mae:.2f}, R²: {r2:.4f}, RMSE: {rmse:.2f}")

evaluate_model(y_test, y_pred_lr, "Regressão Linear")
evaluate_model(y_test, y_pred_rf, "Random Forest Regressor")

# %% Gráfico de comparação entre valores reais e previstos
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Valor Real')
plt.ylabel('Previsão do Modelo (Random Forest)')
plt.title('Avaliação do Modelo: Real vs Predito')
plt.savefig(os.path.join('images', 'real_vs_predito.png'), dpi=300, bbox_inches='tight')
plt.show()

# %% Importância das features no modelo Random Forest
ohe_features = model_rf.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(cat_features)
all_features = num_features + list(ohe_features)

# Extraindo importâncias
importances = model_rf.named_steps['regressor'].feature_importances_
df_imp = pd.DataFrame({'feature': all_features, 'importance': importances})
df_imp = df_imp.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=df_imp, palette='viridis')
plt.title('Importância das variáveis no Modelo Random Forest')
plt.savefig(os.path.join('images', 'importancia_variaveis_random_forest.png'), dpi=300, bbox_inches='tight')
plt.show()

# %%
