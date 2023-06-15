from email import header
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# dataset = pd.read_csv('./data.csv')


# x = dataset.iloc[:, [2, 3]].values
# y = dataset.iloc[:, 4].values
# from sklearn.model_selection import train_test_split
# xtrain, xtest, ytrain, ytest = train_test_split(
#         x, y, test_size = 0.25, random_state = 0)
# from sklearn.preprocessing import StandardScaler
# sc_x = StandardScaler()
# xtrain = sc_x.fit_transform(xtrain)
# xtest = sc_x.transform(xtest)
# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(random_state = 0)
# classifier.fit(xtrain, ytrain)
st.subheader("VISUALISATION OF SIMPLE MOVING AVERAGE TRADING STRATEGY FOR DIFFERENT PERIOD ON RELIANCE")




import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import math

pd.set_option('mode.chained_assignment', None)


# Initialise the data 
long_MA = 200
short_MA = 17
initial_wealth = '1000'
stock = 'RELIANCE.NS'
period = '60d'
start_date =  '2015-01-01'
end_date = '2020-12-31'
interval = '1d'
totalprofit = 0


def get_stock_data(stock,startdate,enddate,period,interval):
        ticker = stock  
        yf.pdr_override()
        df = yf.download(tickers=stock, start=startdate, end=enddate, interval=interval,period=period)
        df.reset_index(inplace=True) 
        df['date'] = df['Date'].dt.date
      
        return df


def ma_strategy(df,short_MA,long_MA):
    df['long_MA'] = df['Close'].rolling(int(long_MA)).mean()
    df['short_MA'] = df['Close'].rolling(int(short_MA)).mean()
    df['crosszero'] = np.where(df['short_MA'] < df['long_MA'], 1.0, 0.0)
    df['position'] = df['crosszero'].diff()
    df['position'].iloc[-1] = -1
    for i, row in df.iterrows():
        if df.loc[i,'position'] == 1 :
                buy_price = round(df.loc[i,'Close'],2)
                df.loc[i,'buy'] = buy_price
        if df.loc[i,'position'] == -1 :
                sell_price = round(df.loc[i,'Close'],2)
                df.loc[i,'sell'] = sell_price
    return df

def ema_strategy(df,short_MA,long_MA):
    df['short_MA'] = df['Close'].ewm(span = short_MA).mean()
    df['long_MA'] = df['Close'].ewm(span = long_MA).mean()
    df['crosszero'] = np.where(df['short_MA'] < df['long_MA'], 1.0, 0.0)
    df['position'] = df['crosszero'].diff()
    df['position'].iloc[-1] = -1
    for i, row in df.iterrows():
        if df.loc[i,'position'] == 1 :
                buy_price = round(df.loc[i,'Close'],2)
                df.loc[i,'buy'] = buy_price
        if df.loc[i,'position'] == -1 :
                sell_price = round(df.loc[i,'Close'],2)
                df.loc[i,'sell'] = sell_price
    
    return df


def buy_sell_signals(df,stock,start_date,end_date):
    
        totalprofit = 0
        print('Stock: {}'.format(stock))
        print('Period: {} - {}'.format(start_date, end_date))
        print('-'*67)
        print('{:^7}{:^10}{:^15}{:^10}{:^15}'.format('S/N','Buy Date','Buy Price($)','Sell Date','Sell Price($)'))
        print('-'*67)

        for i, row in df.iterrows():
                        if df.loc[i,'position'] == 1 :
                                buy_price = round(df.loc[i,'buy'],2)
                                buydate = df.loc[i,'Date']
                        if df.loc[i,'position'] == -1 :
                                sell_price = round(df.loc[i,'sell'],2)
                                selldate  = df.loc[i,'Date']
                                profit = sell_price - buy_price
                                profit = round(profit,2)
                                totalprofit = totalprofit + profit
                                totalprofit = round(totalprofit,2)
                                print('{:^7}{}{:^15}{}{:^15}'.format(i,buydate,buy_price,selldate,sell_price))
                        
        print('')
        print('')
        return df

def backtest(df,stock,startdate,enddate,initial_wealth) :
        # assumptions:
        initial_wealth = int(initial_wealth)
        profitloss = 0 
        position = 0
        total_profit = 0 
        qty = 0
        balance = initial_wealth
        buy_p = 0 # per share 
        total_buy_p = 0
        total_sell_p = 0 
        MA_wealth = initial_wealth # moving average wealth
        LT_wealth = initial_wealth # long-term wealth
        inital_sell = 0 
        df['position'].iloc[-1] = -1
                
    

        print('Stock: {}'.format(stock))
        print('Period: {} - {}'.format(startdate, enddate))
        print('Initial Wealth: {}'.format(initial_wealth))
        print('-'*100)
        print('{:^7}{:^15}{:^10}{:^15}{:^20}{:^20}{:^10}{:^20}{:^20}{:^20}{:^20}'.format('Sr. No','Buy Date','Buy Price($)','Sell Date','Sell Price($)','Investment($)','Qty','total_buy_p','total_sell_p','profitloss','MA_wealth'))
                                                                              
        print('-'*100)
        for i,row in df.iterrows():
            if position == 0:
                if df.loc[i,'position'] == 1:
                    buy_p =round( df.loc[i,'Close'],2)
                    buy_d = df.loc[i,'Date']
                    balance = balance + total_sell_p
                    ibalance = balance
                    qty = balance / buy_p
                    qty = math.trunc(qty)
                    total_buy_p = round(buy_p * qty,2)
                    balance = balance - total_buy_p 
                    position = 1       
                else:
                    price = df.loc[i,'Close'] 
                    if qty == 0 and MA_wealth == initial_wealth:

                        df.loc[i,'MA_wealth'] = MA_wealth
                    elif qty != 0 and MA_wealth != initial_wealth:
                        
                        df.loc[i,'MA_wealth'] = MA_wealth 
            elif position == 1:
                if df.loc[i,'position'] == -1:
                    sell_p = round(df.loc[i,'Close'],2)
                    sell_d = df.loc[i,'Date']
                  
                    total_sell_p = round(sell_p * qty,2)
                    profitloss = round(total_sell_p - total_buy_p,2)
                    total_profit = round(total_profit + profitloss,2)
                    sell_balance = round(balance + total_profit,2)
                    # MA_wealth = round(balance + total_sell_p,2)
                    balance = round(balance,2)
                    
                    print('{:^7}{}{:^15}{}{:^15}{:^15}{:^15}{:^15}{:^20}{:^15}{:^10}'.format(i,buy_d,buy_p,sell_d,sell_p,ibalance,qty,total_buy_p,total_sell_p,profitloss,MA_wealth ))
                  
                    sell_balance = balance + total_sell_p
                    MA_wealth = round(balance + total_sell_p,2)
                    position = 0
                else:
                    price = df.loc[i,'Close'] 
                    stockprice = price * qty
                    MA_wealth = balance + stockprice
                    df.loc[i,'MA_wealth'] = MA_wealth
                    # print(MA_wealth)

            # long-term strategy           
        first_date = df['Date'].iloc[0]  
        initial_price = df['Close'].iloc[0]
        qty = LT_wealth/initial_price

        for i,row in df.iterrows():
            df.loc[i,'LT_wealth'] = df.loc[i,'Close'] * qty
                    
        last_date = df['Date'].iloc[-1]
        final_price = df['Close'].iloc[-1]
        
        LT_buy_p = initial_price * qty
        LT_sell_p = final_price * qty
        LT_profitloss = LT_sell_p - initial_wealth
        LT_wealth = initial_wealth + LT_profitloss
        MA_profitloss = MA_wealth - initial_wealth    
        MA_profitloss = round(MA_profitloss,2)
        LT_profitloss = round(LT_profitloss,2)


        print('-'*100)
        print('Short MA Profit/Loss: ${:,}, Long MA Profit/Loss: ${:,}'.format(MA_profitloss,LT_profitloss))
        print('')
        print('Short MA Final Wealth: ${:,.2f}, Long MA Final Wealth: ${:,.2f}'.format(MA_wealth,LT_wealth))
        print('-'*100)

        return df



def graph(df,stock):
    fig = plt.figure(figsize = [15,6])
    ax = plt.subplot(1,1,1)
    ax.plot(df['Date'], df['long_MA'], label = 'long MA')
    ax.plot(df['Date'], df['short_MA'], color = 'orange', label = 'short MA')
    ax.plot(df['Date'], df['Close'], color = 'black', label = 'Close')
    ax.plot(df['Date'], df['buy'], color = 'green', label = 'Buy',marker = '^')
   
    ax.plot(df['Date'], df['sell'], color = 'red', label = 'Sell',marker = 'v')
    ax.legend(loc = 'upper right')
    ax.set_xlabel('Date')
    ax.set_title(stock)
    plt.show()

    fig = plt.figure(figsize = [15,6])
    ax = plt.subplot(1,1,1)
    ax.plot(df['Date'], df['MA_wealth'], color = 'black', label = 'MA strategy wealth')
    ax.plot(df['Date'], df['LT_wealth'], color = 'red', label = 'buy and hold wealth')
    ax.legend(loc = 'upper left')
    ax.set_xlabel('date')
    ax.set_title(stock)
    plt.show()



print(stock)

df = get_stock_data(stock,start_date,end_date,period,interval)
df = ma_strategy(df,long_MA,short_MA)
df = buy_sell_signals(df,stock,start_date,end_date)
df = backtest(df,stock,start_date,end_date, initial_wealth)
graph(df,stock)



# threshold = st.slider('Threshold', 0.0, 1.0,0.5)
# y_pred = (classifier.predict_proba(xtest)[:,1] >= threshold).astype(bool)
# from sklearn.metrics import confusion_matrix
# fig, ax = plt.subplots()
# cm = confusion_matrix(ytest, y_pred)
# cm_matrix = pd.DataFrame(data=cm, columns=['Predicted Negative', 'Predicted Positive'],index=['Actual Negative', 'Actual Positive'])

# sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu',ax=ax)
# if threshold<0.5 :
#         plt.arrow( 0.70, 0.5, 0.5, 0.0,fc="r", ec="r",head_width=0.05, head_length=0.1 )
#         ax.arrow( 0.70, 1.5, 0.5, 0.0,fc="r", ec="r",head_width=0.05, head_length=0.1 )
# elif threshold>0.5 :
#         ax.arrow( 1.3, 0.5, -0.5, 0.0,fc="r", ec="r",head_width=0.05, head_length=0.1 )
#         ax.arrow( 1.3, 1.5, -0.5, 0.0,fc="r", ec="r",head_width=0.05, head_length=0.1 )
# plt.text(0.425, 0.3,"TN", fontsize=18,color='red')
# plt.text(1.425, 0.3,"FP", fontsize=18,color='red')
# plt.text(0.425, 1.3,"FN", fontsize=18,color='red')
# plt.text(1.425, 1.3,"TP", fontsize=18,color='red')

# fig2, ax = plt.subplots()
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve
# logit_roc_auc = roc_auc_score(ytest, classifier.predict_proba(xtest)[:,1])
# fpr, tpr, thresholds = roc_curve(ytest, classifier.predict_proba(xtest)[:,1])
# # plt.figure()
# TN = cm[0][0]
# FN = cm[1][0]
# TP = cm[1][1]
# FP = cm[0][1]
# tp = TP/(TP+FN)
# fp = FP/(FP+TN)
# ax.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
# plt.plot(fp, tp, marker="o", markersize=8, markeredgecolor="black", markerfacecolor="red")
# ax.plot([0, 1], [0, 1],'r--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.legend(loc="lower right")
# col1, col2 = st.columns(2)
# col1.header("Confusion Matrix")
# col1.write(fig)
# col2.header("ROC-AUC")
# col2.write(fig2)

# col1.header("TPR= TP/(TP+FN)")
# col1.header("FPR= FP/(FP+TN)")
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