# Your OWN AI — Java VectorDB + RAG

Java/Spring Boot port of the original **Your-OWN-AI / VectorDB** project.

## What is implemented

- HNSW approximate nearest-neighbor index from scratch
- KD-Tree nearest-neighbor index
- Brute-force KNN baseline
- Cosine, Euclidean and Manhattan distance
- REST API
- Document chunking
- Ollama embeddings using `nomic-embed-text`
- Local LLM generation using `llama3.2`
- End-to-end local RAG pipeline
- HNSW graph statistics
- Interactive search/benchmark web UI
- Java 17 + Spring Boot

## Architecture

```text
Browser
   |
   v
Spring Boot REST API
   |
   +--> VectorDatabase
   |       +--> BruteForceIndex
   |       +--> KdTreeIndex
   |       +--> HnswIndex
   |
   +--> DocumentDatabase
   |       +--> HNSW
   |       +--> semantic retrieval
   |
   +--> OllamaClient
           +--> nomic-embed-text
           +--> llama3.2
```

## Run

Requirements:

- Java 17+
- Maven
- Ollama

Pull models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

Start the application:

```bash
./mvnw spring-boot:run
```

or:

```bash
mvn spring-boot:run
```

Open:

```text
http://localhost:8080
```

## API

### Demo vectors

```text
GET  /search?v=<16 floats>&k=5&metric=cosine&algo=hnsw
POST /insert
DELETE /delete/{id}
GET  /items
GET  /benchmark
GET  /hnsw-info
GET  /stats
```

### RAG

```text
POST /doc/insert
GET  /doc/list
DELETE /doc/delete/{id}
POST /doc/search
POST /doc/ask
GET  /doc/status
```

Example:

```json
{
  "title": "Operating Systems Notes",
  "text": "A process is a program in execution..."
}
```

Then:

```json
{
  "question": "What is a process?",
  "k": 3
}
```

## Important

This is a **Java port**, not a literal line-by-line translation. C++ classes were redesigned into Java services/components so that the project follows normal Spring Boot structure.

The vector database is intentionally in-memory, matching the original project's behavior. This keeps the implementation focused on indexing, retrieval and RAG flow. For a production-grade version, add persistent storage, tests, Docker, authentication, configurable HNSW parameters, and large-scale ANN benchmarks.

## Development Notes

- Java 17 target with Spring Boot 3.5.6
- Local LLM integration through Ollama
- Vector indexes are implemented in-memory
