import matplotlib.pyplot as plt
from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression

df['Month']=pd.to_datetime(df['Month'])
index=df.set_index('Month').index

dp=DeterministicProcess(index=index,order=1)
X=dp.in_sample()
fX=dp.out_of_sample(steps=5)
y=df['#Passengers']

model=LinearRegression()
model.fit(X,y)
y_pred=model.predict(X)
y_pred_series=pd.Series(y_pred,index=index)

pred=model.predict(fX)
predd=pd.Series(pred,index=fX.index)

fig,ax=plt.subplots()
ax.plot(index,y)
ax.plot(index,y_pred_series)
ax.plot(predd.index,predd)
ax.set_xlabel('Year')
ax.set_ylabel('Passengers')
ax.legend()
