import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Sidebar for input features
st.sidebar.header("User Input Features")
sepal_length = st.sidebar.slider("Sepal length", float(X[:,0].min()), float(X[:,0].max()))
sepal_width = st.sidebar.slider("Sepal width", float(X[:,1].min()), float(X[:,1].max()))
petal_length = st.sidebar.slider("Petal length", float(X[:,2].min()), float(X[:,2].max()))
petal_width = st.sidebar.slider("Petal width", float(X[:,3].min()), float(X[:,3].max()))

# Model
model = RandomForestClassifier()
model.fit(X, y)
prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

# Output
st.write(f"Predicted Iris species: {iris.target_names[prediction][0]}")
