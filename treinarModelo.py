from sklearn.linear_model import LinearRegression


class TreinarModelo:
    def train_model(self, data):
        X = data[['peso', 'altura', 'horas_de_sono']]
        y = data['glicemia']
        model = LinearRegression()
        model.fit(X, y)
        return model