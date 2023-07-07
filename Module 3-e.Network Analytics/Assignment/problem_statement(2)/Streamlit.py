import pandas as pd
import numpy as np
import streamlit as st 
import networkx as nx
import matplotlib.pyplot as plt
import os
import io
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
    
    # st.write('1st upload facebook data')
    # st.write('2nd upload linkedin data')
    # st.write('3rd upload instgram data')
    
    uploadedFile = st.sidebar.file_uploader("You_Can_Load_Multiplefiles" ,type=['csv','xlsx'],accept_multiple_files=True,key="fileUploader")
       
    # for f in uploadedFile:
    #     st.write(f)
    # data_list = []
    # for f in uploadedFile:
    
    data_list = []
    for f in uploadedFile:
        data_list.append(f)
    st.write(data_list[0:3])
    # uploaded_file = data_list[0].name
    # uploaded_file1 = data_list[1].name
    # st.write(uploaded_file)
    if uploadedFile is not None :
        try:
            if data_list[0].name == 'facebook.csv':
                facebook=pd.read_csv(data_list[0])
                if data_list[1].name == 'linkedin.csv':
                    linkedin=pd.read_csv(data_list[1])
                    if data_list[2].name == 'instagram.csv':
                        instagram = pd.read_csv(data_list[2])
                elif data_list[2].name == 'linkedin.csv':
                    linkedin=pd.read_csv(data_list[2])
                    if data_list[2].name == 'instagram.csv':
                        instagram = pd.read_csv(data_list[2])
                    else:
                        instagram=pd.read_csv(data_list[1])
            else :
                st.write('data not matched')
        except:
            try:
                if data_list[0].name == 'facebook.xlsx':
                    facebook=pd.read_excel(data_list[0])
                    if data_list[1].name == 'linkedin.xlsx':
                        linkedin=pd.read_excel(data_list[1])
                        if data_list[2].name == 'instagram.xlsx':
                            instagram = pd.read_excel(data_list[2])
                    elif data_list[2].name == 'linkedin.xlsx':
                        linkedin=pd.read_excel(data_list[2])
                        if data_list[2].name == 'instagram.xlsx':
                            instagram = pd.read_excel(data_list[2])
                        else:
                            instagram=pd.read_excel(data_list[1])
                else :
                    st.write('data not matched')
            except:
                st.warning('You have only csv and excel format')
    else:
        
        st.sidebar.warning("Upload the new data using CSV or Excel file.")
                
        
        
    
    if st.button("Show_networks"):
        st.subheader(":blue[Circular_Network_for_facebook_data]", anchor=None)
        fig, ax = plt.subplots()
        face=nx.Graph(facebook.values)
        face.edges()
        # nx.draw(face, pos=nx.circular_layout(face, scale=1, center=None, dim=2))
        #gh=nx.circular_layout(face, scale=1, center=None, dim=2)
        nx.draw_circular(face,with_labels=True, ax = fig.add_subplot(111))
        st.pyplot(fig)
        
        st.subheader(":blue[Star_Network_for_linkedin_data]", anchor=None)
        fig, ax = plt.subplots()
        linked=nx.Graph(linkedin.values)
        linked.edges()
        nx.draw(linked,with_labels=True, ax = fig.add_subplot(111))
        st.pyplot(fig)
        
        st.subheader(":blue[Star_Network_for_instagram_data]", anchor=None)
        fig, ax = plt.subplots()
        insta=nx.Graph(instagram.values)
        insta.edges()
        nx.draw(insta,with_labels=True, ax = fig.add_subplot(111))
        st.pyplot(fig)
        
if __name__=='__main__':
    main()



      
      
