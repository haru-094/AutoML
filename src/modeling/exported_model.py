import numpy as np


class ExportedModel:
    """Pickle-friendly prediction artifact that restores original class labels."""

    def __init__(self, pipeline, label_encoder=None):
        self.pipeline = pipeline
        self.label_encoder = label_encoder

    def predict(self, X):
        predictions = self.pipeline.predict(X)
        if self.label_encoder is None:
            return predictions
        return self.label_encoder.inverse_transform(np.asarray(predictions, dtype=int))
