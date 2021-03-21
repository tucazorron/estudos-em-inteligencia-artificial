from sklearn.linear_model import LogisticRegression

def logistic_regression(data, useB):
    global result_data
  
    name = "LR B" if useB else "LR A"

    expected, observed, times = [], [], []

    X, Y = generate_dataset(data, useB, FLOW_INTERVAL, N_STEPS, N_FUTURE)
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

    model = LogisticRegression()

    pointers = split_dataset(len(X), SET_SPLIT, TEST_SPLIT)
  
    for i, j, k in pointers:
        start = time.time()
            
        model.fit(X[i:j], Y[i:j])
            
        expected.append(Y[j:k])
        observed.append(model.predict(X[j:k]))
        times.append(time.time() - start)
    
    result_data['results'][name] = evaluate(expected, observed, times, name)