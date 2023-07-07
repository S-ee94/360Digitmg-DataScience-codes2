# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 19:12:51 2021

@author: saira
"""

import pandas as pd
import networkx as nx

facebook=pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 3-e.Network Analytics\Assignment\problem_statement(2)\facebook.csv")
linkedin=pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 3-e.Network Analytics\Assignment\problem_statement(2)\linkedin.csv") 
instagram=pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 3-e.Network Analytics\Assignment\problem_statement(2)\instagram.csv")
facebook.info()


import matplotlib.pyplot as plt
#circular network for facebook data
face=nx.Graph(facebook.values)
face.edges()
nx.draw(face, pos=nx.circular_layout(face, scale=1, center=None, dim=2))
gh=nx.circular_layout(face, scale=1, center=None, dim=2)
nx.draw_circular(face,with_labels=True)
plt.draw_circular()


#star network for linkedin
linkedin.info()
linked=nx.Graph(linkedin.values)
linked.edges()
nx.draw(linked,with_labels=True)

# plt.draw()
#star network for instagram
instagram.info()
insta=nx.Graph(instagram.values)
insta.edges()
nx.draw(insta,with_labels=True)
plt.draw()



