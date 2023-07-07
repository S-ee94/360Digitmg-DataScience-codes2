
# Import Libraries and Dataset
import pandas as pd
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from urllib.parse import quote

user = 'root'
db = 'connecting_routes_db'
pw = 'Seemscrazy1994#'

engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote(f'{pw}'))

G = pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 3-e.Network Analytics\Assignment\problem_statement(1)\connecting_routes.csv")

G.to_sql('connecting_routes', con = engine, if_exists='replace', chunksize = 1000, index =False)

sql = 'select * from connecting_routes;'
G = pd.read_sql_query(sql, con = engine)

# rr = pd.read_csv(r"F:/Ashutosh tasks/routes.csv")s
G.dtypes
# rr.dtypes
G = G.iloc[:501, 0:10]

g = nx.Graph()

g = nx.from_pandas_edgelist(G, source = 'source airport', target = 'destination airport')

print(nx.info(g))

b = nx.degree_centrality(g)  # Degree Centrality
print(b) 

#check which pos is best
pos = nx.spring_layout(g, k = 0.15)
nx.draw_networkx(g, pos, node_size = 25, node_color = 'blue')
#
pos = nx.spring_layout(g, k = 0.25)
nx.draw_networkx(g, pos, node_size = 25, node_color = 'blue')
#
pos = nx.spring_layout(g, k = 0.025)
nx.draw_networkx(g, pos, node_size = 25, node_color = 'blue')
#
pos = nx.spring_layout(g, k = 0.015)
nx.draw_networkx(g, pos, node_size = 25, node_color = 'blue')


data = pd.DataFrame({"closeness":pd.Series(nx.closeness_centrality(g)),
                     "Degree": pd.Series(nx.degree_centrality(g)),
                     "eigenvector": pd.Series(nx.eigenvector_centrality(g)),
                     "betweenness": pd.Series(nx.betweenness_centrality(g)),
                     "cluster_coeff": pd.Series(nx.clustering(g))}) 

# Average clustering
cc = nx.average_clustering(g) 
print(cc)











#######################################################################################################

