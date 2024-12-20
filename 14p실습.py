# 인공지능 기초 수업
import numpy as np
from sklearn.linear_model import LinearRegression

linear_model= LinearRegression()

X = [[168,0],[166,0],[173,0],[165,0],[177,0],[163,0],[178,0],[172,0],
     [163,1],[162,1],[171,1],[162,1],[164,1],[162,1],[158,1],[173,1]]
y = [65,61,68,63,68,61,76,67,
     55,51,59,53,61,56,44,57]


linear_model.fit(X, y) # 학습
# y=mx+b에서

coef = linear_model.coef_ # 기울기 구하기 (m)
intercept = linear_model.intercept_ # 상수 구하기 (b)
score=linear_model.score(X, y) # 오차 계산_내가 만든 기울기 이거 넣었을 때 오차가 얼마에여

print ("y = {}*X + {:.2f}".format(coef.round(2), intercept)) # 소수점 둘째자리에서 반올림
print ("데이터와 선형 회귀 직선의 관계점수 :  {:.1%}".format(score))

'''
import matplotlib.pyplot as plt

plt.scatter(X, y, color='blue', marker='D')
y_pred = linear_model.predict(X)
plt.plot(X, y_pred, 'r:')
plt.show()
'''

unseen = [[167,0], [167,1]] # 167일 때 남자/여자
result = linear_model.predict(unseen) # 학습된 걸 기반으로 새로운 데이터(키 167) 넣었을 때 결과
print ("키 {}cm는 몸무게 {}kg으로 추정됨".format(unseen, result.round(1)))

