package com.shreyash.vectordb.service;

import org.springframework.stereotype.Component;
import java.util.*;

@Component
public class TextChunker {
    public List<String> chunk(String text, int chunkWords, int overlapWords) {
        String[] words = text.trim().split("\\s+");
        if (words.length == 0 || text.isBlank()) return List.of();
        if (words.length <= chunkWords) return List.of(text.trim());

        int step = chunkWords - overlapWords;
        List<String> chunks = new ArrayList<>();

        for (int start = 0; start < words.length; start += step) {
            int end = Math.min(start + chunkWords, words.length);
            chunks.add(String.join(" ", Arrays.copyOfRange(words, start, end)));
            if (end == words.length) break;
        }
        return chunks;
    }
}
