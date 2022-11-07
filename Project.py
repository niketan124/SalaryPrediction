import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.sidebar.title('Navigation')
st.set_option('deprecation.showPyplotGlobalUse', False)

val = st.sidebar.radio("Select the options below", ["Home", "Prediction", "Contribution"])

#--------------------Data Preprocessing-----------#

df = pd.read_csv('./Salary_Data.csv')
x = df.iloc[:, 0:1].values
y = df.iloc[:,-1].values

#----------------------Training Data-----------------------#
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(x, y)
y_pred = lr.predict(x)

#-----------------------#---------------------------#
if val=="Home":
    st.write("# Experience v/s Salary")
    st.image('./web-developer-salary.png')
    
    isShowTable = st.checkbox("Show Table")
    if isShowTable:
        st.dataframe(df, width=800)
    typeOfGraph = st.selectbox("Select the type of graph to visualize the actual data", ["Interactive", "Non-Interactive"])
    
    if typeOfGraph=="Interactive":
        plt.scatter(x, y)
        plt.xlabel("Experience")
        plt.ylabel("Salary")
        st.pyplot()
        
#----------------------#-----------------------#
    
elif val=="Prediction":
    st.write("# Predict salary by using Machine Learning")
    plt.scatter(x, y)
    plt.xlabel("Experience")
    plt.ylabel("Salary")
    plt.plot(x, lr.predict(x), color="red")
    st.pyplot()
    
    st.write("### Your model accuracy")
    plt.plot(y, label="acctual output")
    plt.plot(y_pred, label="predicted outpur")
    plt.legend()
    st.pyplot()
    st.write("### your model accuracy is:")
    st.success(f"{r2_score(y, y_pred)*100:.2f} percentage")
    
    exp = st.number_input("Please enter your experience to predict salary", min_value=1,step=1, value=1)
    isbuttonClicked = st.button("Submit")
    if isbuttonClicked:
        st.success(f"your salary might be {lr.predict([[exp]])[0]:.2f} rupees")
        
        
else:
    st.write("## Could you please contribute to our salary dataset")
    st.write("Please give valid data and spread knowledge ❤️")
    exp = st.number_input("Please enter your experience", min_value=1,step=1)
    salary = st.number_input("Please enter your salary", min_value=1000,step=1000)
    toAdd = {
        "YearOfExperience": [exp],
        "Salary": [salary]
    }
    isClicked = st.button("Contribute")
    if isClicked:
        toAdd = pd.DataFrame(toAdd)
        toAdd.to_csv('./Contribute.csv', mode="a",header=False, index=False)
        st.success("Sucessfully Contributed! 😍")
        da = pd.read_csv('./Contribute.csv')
        st.dataframe(data=da)