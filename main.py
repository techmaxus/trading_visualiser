from email import header
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
st.set_page_config(layout="centered")

dataset = pd.read_csv('./data.csv')
dataset.Gender= [1 if each=="Male" else 0 for each in dataset.Gender]
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)
st.subheader("VISUALISATION OF EFFECT OF CHANGING THRESHOLD IN LOGISTIC REGRESSION")
from sklearn.metrics import confusion_matrix
threshold = st.slider('Threshold', 0.0, 1.0,0.5)
y_t = (classifier.predict_proba(xtest)[:,1] >= 0.5).astype(bool)
ct = confusion_matrix(ytest, y_t)
y_pred = (classifier.predict_proba(xtest)[:,1] >= threshold).astype(bool)


fig, ax = plt.subplots()
cm = confusion_matrix(ytest, y_pred)
c =np.array([["0000","0000"],["0000","0000"]])
if(cm[0][0]>ct[0][0]):
        c[0][0]=str(cm[0][0])+"↑"
if(cm[0][1]>ct[0][1]):
        c[0][1]=str(cm[0][1])+"↑"
if(cm[1][0]>ct[1][0]):
        c[1][0]=str(cm[1][0])+"↑"
if(cm[1][1]>ct[1][1]):
        c[1][1]=str(cm[1][1])+"↑"

if(cm[0][0]<ct[0][0]):
        c[0][0]=str(cm[0][0])+"↓"
if(cm[0][1]<ct[0][1]):
        c[0][1]=str(cm[0][1])+"↓"
if(cm[1][0]<ct[1][0]):
        c[1][0]=str(cm[1][0])+"↓"
if(cm[1][1]<ct[1][1]):
        c[1][1]=str(cm[1][1])+"↓"

if(cm[0][0]==ct[0][0]):
        c[0][0]=str(cm[0][0])
if(cm[0][1]==ct[0][1]):
        c[0][1]=str(cm[0][1])
if(cm[1][0]==ct[1][0]):
        c[1][0]=str(cm[1][0])
if(cm[1][1]==ct[1][1]):
        c[1][1]=str(cm[1][1])

ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)

ax.text(x=0, y=0,s=c[0, 0], va='center', ha='center', size='xx-large')
ax.text(x=1, y=0,s=c[1, 0], va='center', ha='center', size='xx-large')
ax.text(x=0, y=1,s=c[0,1], va='center', ha='center', size='xx-large')
ax.text(x=1, y=1,s=c[1, 1], va='center', ha='center', size='xx-large')
 
plt.xlabel('ACTUAL', fontsize=18)
plt.ylabel('PREDICTED', fontsize=18)
plt.yticks([0, 1], ['1', '0'])
plt.xticks([0, 1], ['1', '0'])





fig2, ax = plt.subplots()
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(ytest, classifier.predict_proba(xtest)[:,1])
fpr, tpr, thresholds = roc_curve(ytest, classifier.predict_proba(xtest)[:,1])
# plt.figure()
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]

tp = TP/(TP+FN)
fp = FP/(FP+TN)
ax.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot(fp, tp, marker="o", markersize=8, markeredgecolor="black", markerfacecolor="red")
ax.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")



col1, col2, = st.columns(2)


col1.header("Confusion Matrix")
col1.write(fig)

col2.header("ROC-AUC")
col2.write(fig2)




footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}


.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed by <a href="https://github.com/techmaxus" target="_blank">Lakshay Kapoor</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)