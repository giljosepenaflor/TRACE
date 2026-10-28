"""Frozen sentence embeddings and a train-only supervised contrastive projection."""

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from .features import interaction_text


class BehavioralEncoder:
    def __init__(self, kind="sentence-transformer", model_name="sentence-transformers/all-MiniLM-L6-v2",
                 local_only=True, representation_mode='prompt_response'):
        self.kind = kind
        self.model_name = model_name
        self.local_only = local_only
        self.representation_mode = representation_mode
        self.projection = None
        self.training_info = None
        self._model = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def encode_base(self, records):
        texts = [interaction_text(r, getattr(self, 'representation_mode', 'prompt_response')) for r in records]
        if self.kind == "demo-hashing":
            return HashingVectorizer(n_features=128, alternate_sign=False,
                                     ngram_range=(1, 2)).transform(texts).toarray()
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device="cpu",
                                              local_files_only=self.local_only)
        return np.asarray(self._model.encode(texts, normalize_embeddings=True,
                                             show_progress_bar=False))

    def fit_transform(self, records, labels, epochs=60):
        base = self.encode_base(records)
        if self.kind == "demo-hashing":
            return base
        import torch
        import torch.nn.functional as functional
        torch.manual_seed(42)
        x = torch.tensor(base, dtype=torch.float32)
        y = torch.tensor(labels)
        # A small learned projection keeps the pretrained sentence encoder frozen.
        projection = torch.nn.Linear(base.shape[1], 64, bias=False)
        optimizer = torch.optim.Adam(projection.parameters(), lr=0.003)
        diagonal = torch.eye(len(x), dtype=torch.bool)
        positives = y[:, None].eq(y[None, :]) & ~diagonal
        valid = positives.sum(dim=1) > 0
        if not bool(valid.all()):
            raise ValueError("Contrastive training requires at least two training examples per class.")
        for _ in range(epochs):
            z = functional.normalize(projection(x), dim=1)
            scores = z @ z.T / 0.1
            scores = scores.masked_fill(diagonal, -1e9)
            log_probabilities = scores - torch.logsumexp(scores, dim=1, keepdim=True)
            loss = -((log_probabilities * positives).sum(dim=1) / positives.sum(dim=1)).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError('Contrastive loss is nonfinite.')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.projection = projection.weight.detach().numpy().T.copy()
        self.training_info = {'epochs': epochs, 'input_dimensions': int(base.shape[1]),
                              'output_dimensions': 64, 'frozen_sentence_encoder': True,
                              'final_contrastive_loss': float(loss.detach()),
                              'fit_sample_ids': [r['sample_id'] for r in records]}
        return self._project(base)

    def _project(self, base):
        if self.projection is None:
            return base
        vectors = base @ self.projection
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    def transform(self, records):
        return self._project(self.encode_base(records))
