# pip3 install apyori

# importacao das bibliotecas
import pandas as pd
import numpy as np
from apyori import apriori
from matplotlib import pyplot as plt

# Configurando print das colunas e linhas
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def apriori_alg(observations):
    # Aplicacao do algoritmo apriori
    association = apriori(observations, min_support=0.1)# , min_lift=1.2, min_length=2)
    association_results = list(association)

    # Criando dataframe para visualizacao dos dados
    df = pd.DataFrame(columns=('Names','Support','Confidence','Lift'))

    Support =[]
    Confidence = []
    Lift = []
    Items = []

    for RelationRecord in association_results:
        for ordered_stat in RelationRecord.ordered_statistics:
            Support.append(RelationRecord.support)
            Items.append(RelationRecord.items)
            Confidence.append(ordered_stat.confidence)
            Lift.append(ordered_stat.lift)

    df['Names'] = list(map(set, Items))                                   
    df['Support'] = Support
    df['Confidence'] = Confidence
    df['Lift']= Lift

    return df.copy()

# Selecionando dados a partir do csv
df = pd.read_csv('_ASSOC_VoleiStars.csv', index_col=None, encoding='iso-8859-1')

# Separando os jogadores por colunas
players = df['Jogadore(a)s'].str.split(", ", expand=True)
result = df['Resultado'].copy()

# Substitui os pontos de pergunta por A
players[0] = players[0].str.replace('?', 'A').copy()
players[1] = players[1].str.replace('?', 'A').copy()
players[2] = players[2].str.replace('?', 'A').copy()

# Aplica caracter simples em todas as palavras
players[0] = players[0].str.lower().copy()
players[1] = players[1].str.lower().copy()
players[2] = players[2].str.lower().copy()
result = result.str.lower().copy()

# Remove espacos e caracteres especiais restantes
players.replace({r'[^\x00-\x7F]+':'a'}, regex=True, inplace=True)

# Normalizacao dos dados (letras maiusculas)
players[0].apply(lambda x: x.strip().capitalize())
players[1].apply(lambda x: x.strip().capitalize())
players[2].fillna('unknown', inplace=True)
players[2].apply(lambda x: x.strip().capitalize())
result.apply(lambda x: x.strip().capitalize())
players[2] = players[2].replace('unknown', np.nan).copy()

# Dividindo o dataframe entre vitorias e derrotas
partidas_ganhas = players.where(result == "ganhou")
partidas_perdidas = players.where(result == "perdeu")
partidas_ganhas.dropna(how='all', inplace=True)
partidas_perdidas.dropna(how='all', inplace=True)

# Transformando o dataframe em um array de listas
melhores_grupos = [] 
for i in range(0, len(partidas_ganhas)):
    melhores_grupos.append([str(partidas_ganhas.values[i,j]) for j in range(0, 3)])
piores_grupos = [] 
for i in range(0, len(partidas_perdidas)):
    piores_grupos.append([str(partidas_perdidas.values[i,j]) for j in range(0, 3)])

# Chamando funcao apriori
melhores = apriori_alg(melhores_grupos)
piores = apriori_alg(piores_grupos)

# Calculando melhores e piores jogadores (placar)
cpio = piores[piores['Lift'] == 1.0].sort_values(by='Names', ascending=True).copy()
cmel = melhores[melhores['Lift'] == 1.0].sort_values(by='Names', ascending=True).copy()
placar = pd.DataFrame(columns=['Nomes', 'Media'])
placar['Nomes'] = cmel['Names']
placar['Media'] = cmel['Confidence']/cpio['Confidence']
placar = placar[placar.Nomes != set(["nan"])].copy()
placar.sort_values(by='Media', ascending=False, inplace=True)
placar.reset_index(inplace=True)
placar.drop('index', axis=1, inplace=True)

# Mostrando os dados
print ("\nMelhor Grupo: ")
print (melhores[melhores['Confidence'] > 0.62])
print ("\nPior Grupo: ")
print (piores[piores['Confidence'] > 0.62])
print ("\nOrdem dos melhores jogadores: ")
print (placar)