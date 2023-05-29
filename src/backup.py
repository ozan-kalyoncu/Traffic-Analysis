membership = Membership()

shuffled_df = membership.df.sample(frac=1, ignore_index=True)

df = pd.get_dummies(shuffled_df, columns=["day"], drop_first=True)

inputs = df.drop(['degree'], axis=1)

outputs = df['degree']

x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, train_size=0.6, random_state=2)


scaler = MinMaxScaler()

scaler.fit(x_train)

x_train_s = scaler.transform(x_train)
x_test_s = scaler.transform(x_test)

nn = MLPRegressor(hidden_layer_sizes=(100,100,), activation="logistic", max_iter=200, solver="lbfgs", alpha=0.0005, max_fun=500)

nn.fit(x_train_s, y_train)

predicts = nn.predict(x_train_s)


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))

axes[0, 0].plot(x_train["hour"], y_train, "o", color="blue")
axes[0, 0].plot(x_train["hour"], predicts, "o", color="red")
axes[0, 0].grid(True)

plt.tight_layout()
plt.savefig('assets/model.png')

mae = metrics.mean_absolute_error(y_train,predicts)
mse = metrics.mean_squared_error(y_train,predicts)
rsq = metrics.r2_score(y_train,predicts)

print(mae, mse, rsq)

print(nn.score(x_test_s, y_test))
