import unittest

from wine import FEATURE_COLUMNS, predict_quality, load_or_train_model


class WineModelTests(unittest.TestCase):
    def test_prediction_returns_integer_quality(self):
        model = load_or_train_model()
        sample = {column: 0.0 for column in FEATURE_COLUMNS}
        sample["fixed acidity"] = 7.4
        sample["volatile acidity"] = 0.7
        sample["citric acid"] = 0.0
        sample["residual sugar"] = 1.9
        sample["chlorides"] = 0.076
        sample["free sulfur dioxide"] = 11.0
        sample["total sulfur dioxide"] = 34.0
        sample["density"] = 0.9978
        sample["pH"] = 3.51
        sample["sulphates"] = 0.56
        sample["alcohol"] = 9.4

        prediction = predict_quality(model, sample)

        self.assertIsInstance(prediction, int)
        self.assertGreaterEqual(prediction, 3)
        self.assertLessEqual(prediction, 9)


if __name__ == "__main__":
    unittest.main()
