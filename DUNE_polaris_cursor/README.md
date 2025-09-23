# DUNE-GPT

A Retrieval-Augmented Generation (RAG) based LLM application for DUNE scientific documentation. This application integrates DUNE internal databases (DocDB and Indico) with a semantic search system powered by FAISS and an LLM via the Argo API.

## Features

- **Document Integration**: Automatically extracts and processes documents from DUNE DocDB and Indico
- **Semantic Search**: Uses FAISS for fast vector similarity search with sentence transformers
- **LLM Integration**: Leverages Argo API for intelligent question answering
- **Web Interface**: Clean, responsive web UI for easy interaction
- **Fermilab Authentication**: Secure OAuth2 authentication via Fermilab PingFederate
- **CLI Tools**: Command-line interface for management and batch operations
- **Production Ready**: Docker containerization, health checks, and monitoring
- **RESTful API**: Programmatic access via REST endpoints

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   DocDB/Indico  │    │   FAISS Index   │    │   Argo API      │
│   Documents     │───▶│   Embeddings    │───▶│   LLM Response  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DUNE-GPT Application                         │
│  Web UI + REST API + CLI + Document Processing Pipeline        │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- DUNE DocDB credentials
- Argo API credentials
- Optional: DUNE Indico access key
- Docker (for containerized deployment)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd DUNEGPT_polaris_cursor
   ```

2. **Set up environment**
   ```bash
   # Copy environment template
   cp env.example .env
   
   # Edit .env with your credentials
   nano .env
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

4. **Index documents**
   ```bash
   python cli.py index --docdb-limit 50 --indico-limit 50
   ```

5. **Start the server**
   ```bash
   python cli.py serve
   ```

Visit `http://localhost:8000` to access the web interface.

📋 **For detailed installation instructions, including troubleshooting and advanced configuration, see [INSTALL.md](INSTALL.md)**

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   # Ensure .env file is configured
   docker-compose up --build
   ```

2. **Index documents in container**
   ```bash
   docker-compose exec dune-gpt python cli.py index
   ```

## Configuration

All configuration is managed through environment variables. See `env.example` for all available options.

### Required Variables

- `ARGO_API_USERNAME`: Your Argo API username
- `ARGO_API_KEY`: Your Argo API key
- `DUNE_DOCDB_USERNAME`: DUNE DocDB username
- `DUNE_DOCDB_PASSWORD`: DUNE DocDB password

### Authentication Variables (Required if enabled)

- `ENABLE_AUTHENTICATION`: Enable/disable Fermilab authentication (default: true)
- `FERMILAB_CLIENT_ID`: Your Fermilab OAuth2 client ID
- `FERMILAB_CLIENT_SECRET`: Your Fermilab OAuth2 client secret
- `FERMILAB_SESSION_SECRET`: Secret key for user sessions (change in production)
- `FERMILAB_REDIRECT_URI`: OAuth2 redirect URI (must match Fermilab app config)

### Optional Variables

- `DUNE_INDICO_ACCESS_KEY`: Indico access key (for private events)
- `DOC_LIMIT_DOCDB`: Number of DocDB documents to process (default: 50)
- `DOC_LIMIT_INDICO`: Number of Indico documents to process (default: 50)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `LOG_LEVEL`: Logging level (default: INFO)
- `FERMILAB_SCOPE`: OAuth2 scope (default: "openid email profile")

## Usage

### Web Interface

Access the web interface at `http://localhost:8000`:
- **With Authentication**: Login with your Fermilab credentials via OAuth2
- **Without Authentication**: Set `ENABLE_AUTHENTICATION=false` for open access
- Enter questions about DUNE documents
- Get AI-powered answers with source references
- View document links for further reading

### Authentication Flow

1. Visit `http://localhost:8000`
2. Click "Login with Fermilab" if authentication is enabled
3. Authenticate via Fermilab PingFederate
4. Return to DUNE-GPT with full access
5. Use "Logout" link when finished

### CLI Commands

```bash
# Index documents from DocDB and Indico
python cli.py index [--docdb-limit N] [--indico-limit N]

# Query the indexed documents
python cli.py query "What is the DUNE detector design?"

# Show index statistics
python cli.py stats

# Check system health
python cli.py health

# Start web server
python cli.py serve
```

### REST API

#### Search Documents
```bash
curl "http://localhost:8000/api/search?q=DUNE%20detector&top_k=3"
```

#### Health Check
```bash
curl "http://localhost:8000/api/health"
```

#### Get Statistics
```bash
curl "http://localhost:8000/api/stats"
```

## Project Structure

```
DUNEGPT_polaris_cursor/
├── config.py                 # Configuration management
├── main.py                   # FastAPI web application
├── cli.py                    # Command-line interface
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container definition
├── docker-compose.yml       # Docker composition
├── env.example              # Environment template
│
├── src/                     # Source code
│   ├── api/                 # API clients
│   │   └── argo_client.py   # Argo API client
│   ├── core/                # Core business logic
│   │   └── document_processor.py
│   ├── extractors/          # Document extractors
│   │   ├── base.py         # Base extractor class
│   │   ├── docdb_extractor.py
│   │   └── indico_extractor.py
│   ├── indexing/           # FAISS indexing
│   │   └── faiss_manager.py
│   └── utils/              # Utilities
│       └── logger.py       # Logging setup
│
├── templates/              # HTML templates
│   └── index.html
├── static/                 # Static assets
│   └── images/
└── data/                   # Data storage
    └── faiss/              # FAISS index files
```

## Production Deployment

### Environment Setup

1. **Set production environment variables**:
   ```bash
   export DEBUG=false
   export LOG_LEVEL=INFO
   export HOST=0.0.0.0
   export PORT=8000
   ```

2. **Use a reverse proxy** (nginx, traefik) for SSL termination and load balancing

3. **Set up monitoring** using the health check endpoint: `/api/health`

### Performance Considerations

- **FAISS Index**: Stored persistently in `data/faiss/` directory
- **Memory Usage**: Scales with document count and embedding model size
- **GPU Support**: Use `faiss-gpu` for faster search on GPU-enabled systems
- **Concurrent Requests**: FastAPI handles concurrent requests efficiently

### Security Notes

- Keep API credentials secure and rotate regularly
- Use HTTPS in production
- Consider network restrictions for internal deployment
- Implement rate limiting if needed

## Monitoring

### Health Endpoints

- `/api/health` - System health status
- `/api/stats` - Index statistics
- Docker health checks configured

### Logging

- Structured logging with configurable levels
- Container logs available via `docker-compose logs`
- Application metrics in health endpoint

## Troubleshooting

### Common Issues

1. **Empty index**: Run `python cli.py index` to populate the FAISS index
2. **API errors**: Check Argo API credentials and network connectivity
3. **Memory issues**: Reduce document limits or upgrade system memory
4. **Docker build fails**: Ensure sufficient disk space and network connectivity

### Debug Mode

Enable debug mode for detailed logging:
```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
```

## Contributing

1. Follow the existing code structure and patterns
2. Add tests for new functionality
3. Update documentation for API changes
4. Use type hints for better code quality

## License

[Add your license information here]

## Support

For issues and questions:
- Check the logs: `docker-compose logs dune-gpt`
- Run health check: `python cli.py health`
- Review configuration in `.env` file

---

**DUNE-GPT** - Making DUNE scientific documentation accessible through AI-powered search and retrieval. 