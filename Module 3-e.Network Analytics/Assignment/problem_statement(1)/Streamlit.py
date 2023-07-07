import pandas as pd
import numpy as np
import streamlit as st 
import networkx as nx
from sqlalchemy import create_engine
from urllib.parse import quote
import matplotlib.pyplot as plt

def main():
    
    st.title("Network_Analytics")
    st.sidebar.title("Network_Analytics")

    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Network_Analytics </h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    

    uploadedFile = st.sidebar.file_uploader("Choose a file" ,type=['csv','xlsx'],accept_multiple_files=False,key="fileUploader")
    if uploadedFile is not None :
        try:

            data=pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    data = pd.DataFrame(uploadedFile)
                
    else:
        st.sidebar.warning("Upload the new data using CSV or Excel file.")
    
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here")
    db = st.sidebar.text_input("database", "Type Here")
    

    
    
    if st.button("Predict"):
        engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote(f'{pw}'))
        G = data.iloc[:501, 0:10]
        g = nx.Graph()
        try:
            g = nx.from_pandas_edgelist(G, source = 'source airport', target = 'destination airport')
            st.write("Data is connecting_routes")
        except:
            g = nx.from_pandas_edgelist(G, source = 'IATA_FAA', target = 'ICAO')
            st.write("Data is flight_hault")
         
        data = pd.DataFrame({"closeness":pd.Series(nx.closeness_centrality(g)),
                 "Degree": pd.Series(nx.degree_centrality(g)),
                 "eigenvector": pd.Series(nx.eigenvector_centrality(g)),
                 "betweenness": pd.Series(nx.betweenness_centrality(g)),
                 "cluster_coeff": pd.Series(nx.clustering(g))}) 
        
        data.to_sql('centralities', con = engine, if_exists='replace', chunksize = 1000, index =False)
        
        data = data.sample(n = 25)
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap=True)
        st.table(data.style.background_gradient(cmap=cm).set_precision(2))
        
        st.subheader(":blue[Network]", anchor=None)
        
        fig, ax = plt.subplots()
        pos = nx.spring_layout(g, k = 0.015)
        nx.draw_networkx(g, pos, ax=fig.add_subplot(111), node_size = 25, node_color = 'blue')
        st.pyplot(fig)#bbox_inches='tight'
        
        
        
        
        
if __name__=='__main__':
    main()



      
      
