"""Loaders turn a source (a dataset, a folder) into Documents.

Each loader module sets CHUNKING, the name of the chunker (see chunking.CHUNKERS) its documents should go
through: chunking is a per-loader setting, not a stage every document passes through.
"""
