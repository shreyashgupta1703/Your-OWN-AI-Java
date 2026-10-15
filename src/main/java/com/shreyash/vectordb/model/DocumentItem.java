package com.shreyash.vectordb.model;

public record DocumentItem(
        int id,
        String title,
        String text,
        float[] embedding
) {}
