import numpy as np

class KerasNeuralNetwork:

    def fit(self, X, y):
        import keras
        from keras import layers, models

        keras.utils.set_random_seed(42)
        X = np.asarray(X).astype("float32")
        y = np.asarray(y).astype("float32")

        self.model = models.Sequential([
            layers.Input(shape=(X.shape[1],)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ])
        self.model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.AUC(name="auc")],
        )
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        )
        self.model.fit(
            X,
            y,
            validation_split=0.2,
            epochs=100,
            batch_size=64,
            callbacks=[early_stopping],
            verbose=0,
        )
        return self

    def predict_proba(self, X):
        positive = self.model.predict(np.asarray(X).astype("float32"), verbose=0).ravel()
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
