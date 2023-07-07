
# Import Libraries and Dataset
import pandas as pd
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from urllib.parse import quote

user = 'root'
db = 'flight_hault_db'
pw = 'Seemscrazy1994#'

engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote(f'{pw}'))

G = pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 3-e.Network Analytics\Assignment\problem_statement(1)\flight_hault.csv")

G.to_sql('flight_hault', con = engine, if_exists='replace', chunksize = 1000, index =False)

sql = 'select * from flight_hault;'
G = pd.read_sql_query(sql, con = engine)

G.dtypes

G.isnull().sum()

G = G.dropna(axis=0)
G.isnull().sum()

G = G.iloc[:501, 0:12]

g = nx.Graph()

g = nx.from_pandas_edgelist(G, source = 'IATA_FAA', target = 'ICAO')

print(nx.info(g))

b = nx.degree_centrality(g)  # Degree Centrality
print(b) 

pos = nx.spring_layout(g, k = 0.15)
nx.draw_networkx(g, pos, node_size = 25, node_color = 'blue')

data = pd.DataFrame({"closeness":pd.Series(nx.closeness_centrality(g)),# closeness_centrality
                     "Degree": pd.Series(nx.degree_centrality(g)),#degree_centrality
                     "eigenvector": pd.Series(nx.eigenvector_centrality(g)),#eigenvector_centrality
                     "betweenness": pd.Series(nx.betweenness_centrality(g)),#betweenness_centrality
                     "cluster_coeff": pd.Series(nx.clustering(g))}) #cluster_coeff

# Average clustering
cc = nx.average_clustering(g) 
print(cc)
