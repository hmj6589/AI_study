from sklearn.linear_model import LogisticRegression

# 제공된 데이터
X = [[168,0],[166,0],[173,0],[165,0],[177,0],[163,0],[178,0],[172,0],
     [163,1],[162,1],[171,1],[162,1],[164,1],[162,1],[158,1],[173,1]]
y = [65,61,68,63,68,61,76,67,55,51,59,53,61,56,44,57]

y_binary = [1 if weight >= 60 else 0 for weight in y]

# 로지스틱 회귀
logistic_model = LogisticRegression()
logistic_model.fit(X, y_binary)

# 계수와 절편, 점수
print('계수:', logistic_model.coef_)
print('절편:', logistic_model.intercept_)
print('점수:', logistic_model.score(X, y_binary))

testX = [ [167,0], [167,1]] # 처음 본 데이터 (test data)
# 예측 확률
y_pred = logistic_model.predict_proba(testX)
print('예측 확률:', y_pred)

# 예측 결과
y_pred_logistic = logistic_model.predict(testX)
print('예측 결과:', y_pred_logistic)

# 실행 결과
# 계수: [[ 0.12397411 -1.58449714]]
# 절편: [-19.58406348]
# 점수: 0.8125
# 예측 확률: [[0.24608324 0.75391676] -> 60을 넘을 확률 / 안넘을 확률 ????
#  [0.6141724  0.3858276 ]]
# 예측 결과: [1 0]