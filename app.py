import pandas as pd
import streamlit as st
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , classification_report



st.title("Spam Email Detector")
st.write("Check if an email is spam or not ")

#cleaning the data
def clean_text(text):
    text=text.lower()
    text=re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

#load and train cache

@st.cache_resource
def load_and_train():
    df=pd.read_csv("sms.tsv" , sep='\t' , names=['label' , 'message'])
    df['label']=df['label'].map({'ham':0 , 'spam':1})
    df['message']=df['message'].apply(clean_text)

    vectorizer = TfidfVectorizer(
    stop_words='english',      # remove common words
    max_features=3000,         # limit features
    ngram_range=(1, 2)         # unigrams + bigrams 🔥
    )
    X=vectorizer.fit_transform(df['message'])
    Y=df['label']

    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

    model=LogisticRegression(max_iter=1000)
    model.fit(X_train,Y_train)
    Y_pred=model.predict(X_test)
    accuracy=accuracy_score(Y_test,Y_pred)
    classification=classification_report(Y_test,Y_pred)
    return model,vectorizer,accuracy,classification

model,vectorizer,accuracy,classification=load_and_train()

#input from user
email=st.text_area("Enter the email content here:")
if st.button("Check"):
    if email.strip()=="":
        st.warning("Please enter an email content.")
    else:
        cleaned=clean_text(email)
        vec=vectorizer.transform([cleaned])
        pred=model.predict(vec)
        prob=model.predict_proba(vec)

        spam_probability=prob[0][1]
        if pred[0]==1:
           st.error(f"🚨 Spam Detected (Confidence: {spam_probability:.2f})")
        else:
            st.success(f"✅ Not Spam (Confidence: {1-spam_probability:.2f})")

#evaluation 

        st.subheader("Model Evaluation")
        st.write(f"Accuracy: {accuracy:.2f}")

        st.subheader("Classification Report")
        st.write(classification)
