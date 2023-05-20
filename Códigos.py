
- Setup:

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
import graphviz 
from sklearn import tree
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix

- Criando o DataFrame:

data = pd.read_csv('dataset.csv', sep=';')
empresa = data[['Segmento', 'Pais', 'Total_Vendas']]

- Explorando o Dataframe:

empresa["Total_Vendas"] = empresa["Total_Vendas"].str.replace(',','.')

empresa["Total_Vendas"] = pd.to_numeric(empresa["Total_Vendas"])

empresa[['Segmento']].drop_duplicates()

empresa.dropna()

empresa['Pais_num'] = empresa['Pais'].apply(lambda Pais: 1 if Pais == 'Brazil' else 0) 

empresa_limpo = empresa[['Pais_num', 'Total_Vendas', 'Segmento']]

empresa_limpo = empresa.loc[empresa['Pais']=='Brazil'].drop_duplicates()

empresa_limpo = empresa_limpo.drop(columns=['Pais'])

- Visualização do DataFrame:

with sns.axes_style('whitegrid'):
  grafico = sns.pairplot(data=empresa_limpo, hue="Segmento", palette="pastel")

  grafico1 = sns.barplot(data=empresa_limpo, x='Segmento', y='Total_Vendas')
grafico1.set(title='Preço de Venda por Segmento', xlabel='Segmento', ylabel='Total de Vendas')

grafico2 = sns.lineplot(data=empresa_limpo, x='Total_Vendas', y='Segmento', hue='Segmento')
grafico2.set(title='Segmento de Venda por Região', xlabel='Segmento das vendas', ylabel='Estado')

Corporativo = empresa_limpo.loc[empresa_limpo["Segmento"]=="Corporativo"].agg('sum')
print(Corporativo)

home_office = empresa_limpo.loc[empresa_limpo["Segmento"]=="Home Office"].agg('sum')
print(home_office)

Consumidor = empresa_limpo.loc[empresa_limpo["Segmento"]=="Consumidor"].agg('sum')
print(Consumidor)

- Treino / Teste:

predictors_train, predictors_test, target_train, target_test = train_test_split(
    empresa_limpo.drop(['Segmento'], axis = 1),
    empresa_limpo['Segmento'],
    test_size=0.3,
    random_state=123
)

predictors_train.head()

predictors_train.shape

predictors_test.head()

target_test.head()

target_test.shape

 - Modelagem:

model = DecisionTreeClassifier()

model = model.fit(predictors_train, target_train)

model.__dict__

tree_data = tree.export_graphviz(model, out_file=None) 
graph = graphviz.Source(tree_data)
graph

- Avaliação do Modelo:

target_predicted = model.predict(predictors_test)

confusion_matrix = confusion_matrix(target_test, target_predicted)
print(confusion_matrix)

total = confusion_matrix.sum()
print(total)
acertos = np.diag(confusion_matrix).sum()
print(acertos)
acuracia = acertos / total 
print(acuracia)
print(f"{round(100 * acuracia, 2)}%")

- Predição:

empresa_limpo.head(1)

features = np.array([3500, 1]) 
prediction = model.predict(features.reshape(1, -1)) 
print(prediction)



