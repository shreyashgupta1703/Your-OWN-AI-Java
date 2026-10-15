package com.shreyash.vectordb.core;

@FunctionalInterface
public interface Distance {
    float calculate(float[] a, float[] b);
}
