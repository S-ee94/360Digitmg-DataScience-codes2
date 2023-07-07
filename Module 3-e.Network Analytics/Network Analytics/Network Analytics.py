# # Problem Statement: -
# There are two datasetsconsisting of information for the connecting routes and flight halt. Create network analytics models on both the datasets separately and measure degree centrality, degree of closeness centrality, and degree of in-between centrality.
# ●	Create a network using edge list matrix(directed only).
# ●	Columns to be used in R:
# 
# Flight_halt=c("ID","Name","City","Country","IATA_FAA","ICAO","Latitude","Longitude","Altitude","Time","DST","Tz database time")
# 
# connecting routes=c("flights", " ID", "main Airport”, “main Airport ID", "Destination ","Destination  ID","haults","machinary")
# 

# # network analytics:-
# Network analytics is the application of big data principles and tools to the management and security of data networks.
# 
import pandas as pd
import networkx as nx 
import matplotlib.pyplot as plt

from sqlalchemy import create_engine

# Creating engine which link to SQL via python
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user="root",# user
                               pw="Seemscrazy1994#", # passwrd
                               db="air_routes_db")) #database

# Reading data from loacal drive
connecting_routes = pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 3-e.Network Analytics\Network Analytics\connecting_routes.csv")

connecting_routes.head()

# Loading data into sql database
connecting_routes.to_sql('connecting_routes', con = engine, if_exists = 'replace', chunksize = 1000, index= False)

# Reading data from sql database
sql = 'select * from connecting_routes;'
connecting_routes = pd.read_sql_query(sql, con = engine)

connecting_routes.head()

connecting_routes = connecting_routes.iloc[0:150, 1:8]
connecting_routes.columns


for_g = nx.Graph()
for_g = nx.from_pandas_edgelist(connecting_routes, source = 'source airport', 
                                target = 'destination apirport')


print(nx.info(for_g))

# #  centrality:-
# 
# 
# **Degree centrality** is defined as the number of links incident upon a node (i.e., the number of ties that a node has). ... Indegree is a count of the number of ties directed to the node (head endpoints) and outdegree is the number of ties that the node directs to others (tail endpoints).
# 
# **Eigenvector Centrality** The adjacency matrix allows the connectivity of a node to be expressed in matrix form. So, for non-directed networks, the matrix is symmetric.Eigenvector centrality uses this matrix to compute its largest, most unique eigenvalues.
# 
# **Closeness Centrality** An interpretation of this metric, Centralness.
# 
# **Betweenness centrality** This metric revolves around the idea of counting the number of times a node acts as a bridge.


data = pd.DataFrame({"closeness":pd.Series(nx.closeness_centrality(for_g)),
                     "Degree": pd.Series(nx.degree_centrality(for_g)),
                     "betweenness": pd.Series(nx.betweenness_centrality(for_g))}) 

data

nx.closeness_centrality(for_g)
nx.degree_centrality(for_g)
#nx.eigenvector_centrality(for_g)
eigne_vector = nx.eigenvector_centrality(for_g, max_iter=600)
import networkx as nx
G = nx.path_graph(for_g)
centrality = nx.eigenvector_centrality(for_g, max_iter=600)


# Visual Representation of the Network
for_g = nx.Graph()
for_g = nx.from_pandas_edgelist(connecting_routes, source = 'source airport', target = 'destination apirport')

f = plt.figure()
pos = nx.spring_layout(for_g, k = 0.015)
nx.draw_networkx(for_g, pos, ax=f.add_subplot(111), node_size = 15, node_color = 'red')
plt.show()
#f.savefig("graph.png")




