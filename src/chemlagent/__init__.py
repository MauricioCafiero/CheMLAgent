"""CheMLAgent — an agentic system for building and training ML models for chemistry.

Public, Ollama-callable tools live in :mod:`chemlagent.tools`. The underlying
featurizers, models, and cleaning helpers remain in their original modules
(`fingerprints`, `models`, `pytorch_mlp`, `chemprop_model`,
`descriptor_cleaning`).
"""

__all__ = ["tools"]