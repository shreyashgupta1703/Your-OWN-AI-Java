package com.shreyash.vectordb.model;

public record VectorItem(
        int id,
        String metadata,
        String category,
        float[] embedding
) {}
