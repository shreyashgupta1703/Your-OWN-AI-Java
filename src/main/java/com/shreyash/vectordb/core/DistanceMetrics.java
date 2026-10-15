package com.shreyash.vectordb.core;

public final class DistanceMetrics {
    private DistanceMetrics() {}

    public static float euclidean(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            float d = a[i] - b[i];
            sum += d * d;
        }
        return (float) Math.sqrt(sum);
    }

    public static float cosine(float[] a, float[] b) {
        float dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na < 1e-9 || nb < 1e-9) return 1.0f;
        return 1.0f - dot / ((float)Math.sqrt(na) * (float)Math.sqrt(nb));
    }

    public static float manhattan(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
        return sum;
    }

    public static Distance of(String metric) {
        return switch (metric == null ? "cosine" : metric.toLowerCase()) {
            case "euclidean" -> DistanceMetrics::euclidean;
            case "manhattan" -> DistanceMetrics::manhattan;
            default -> DistanceMetrics::cosine;
        };
    }
}
