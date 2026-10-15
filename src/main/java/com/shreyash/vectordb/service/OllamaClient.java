package com.shreyash.vectordb.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Service;

import java.net.URI;
import java.net.http.*;
import java.time.Duration;
import java.util.*;

@Service
public class OllamaClient {
    private final HttpClient http = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(5)).build();
    private final ObjectMapper mapper = new ObjectMapper();

    private final String host = "http://localhost:11434";
    private final String embedModel = "nomic-embed-text";
    private final String genModel = "llama3.2";

    public boolean isAvailable() {
        try {
            HttpRequest req = HttpRequest.newBuilder(URI.create(host + "/api/tags"))
                    .timeout(Duration.ofSeconds(3)).GET().build();
            return http.send(req, HttpResponse.BodyHandlers.ofString()).statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    public float[] embed(String text) {
        try {
            String body = mapper.writeValueAsString(Map.of(
                    "model", embedModel,
                    "prompt", text
            ));
            HttpRequest req = HttpRequest.newBuilder(URI.create(host + "/api/embeddings"))
                    .timeout(Duration.ofMinutes(2))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();

            String response = http.send(req, HttpResponse.BodyHandlers.ofString()).body();
            JsonNode node = mapper.readTree(response).path("embedding");
            float[] result = new float[node.size()];
            for (int i = 0; i < node.size(); i++) result[i] = (float) node.get(i).asDouble();
            return result;
        } catch (Exception e) {
            return new float[0];
        }
    }

    public String generate(String prompt) {
        try {
            String body = mapper.writeValueAsString(Map.of(
                    "model", genModel,
                    "prompt", prompt,
                    "stream", false
            ));
            HttpRequest req = HttpRequest.newBuilder(URI.create(host + "/api/generate"))
                    .timeout(Duration.ofMinutes(5))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();

            JsonNode response = mapper.readTree(
                    http.send(req, HttpResponse.BodyHandlers.ofString()).body()
            );
            return response.path("response").asText("");
        } catch (Exception e) {
            return "Generation failed: " + e.getMessage();
        }
    }

    public String embedModel() { return embedModel; }
    public String genModel() { return genModel; }
}
