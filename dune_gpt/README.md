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

- Python 3.11+
- DUNE DocDB credentials
- Argo API credentials
- Optional: DUNE Indico access key
- Docker (for containerized deployment)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/DUNE/LLM_for_DUNE.git
   cd LLM_fo_DUNE/dune_gpt
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

   # First check which version of Python you are using. This project requires Python >= 3.11 and < 3.13.
   python --version

   # Python 3.11 is the recommended and primary supported version, and is used in production environments (e.g. FNAL gpvm). Newer Python versions (e.g. 3.13) are not yet supported due to upstream dependency limitations (e.g. greenlet / gevent).
   python3.11 -m venv venv

   # Or, if python3 already points to ≥3.11 and < 3.13**
   python3 -m venv venv
   
   source venv/bin/activate

   ```
   ### Note on local development
   For local development on macOS, it may be convenient to use conda to create a Python 3.11 environment:

   ```
   conda create -n dune-gpt python=3.11
   conda activate dune-gpt

   ```
   
   Then proceed with the rest of the commands
   
   ```
   pip3 install -r requirements.txt
   python3 -m spacy download en_core_web_sm
   python3 update_sqlite.py  # Activate your Python environment (conda or venv), then run this command. 

   ```

   To install tesseract for image support

   ```
   brew install tesseract  # for local installation
   
   # For tesseract package installation on Aurora, see below:
   # Create local directories if not exist
   mkdir -p $HOME/.local/bin $HOME/.local/lib
   # Navigate to a temporary folder
   cd /tmp
   # Download Tesseract source (replace version if needed)
   wget https://github.com/tesseract-ocr/tesseract/archive/refs/tags/5.3.1.tar.gz -O tesseract-5.3.1.tar.gz
   tar -xzf tesseract-5.3.1.tar.gz
   cd tesseract-5.3.1
   # Ensure your shell can find the Tesseract binary:
   export PATH=$HOME/.local/bin:$PATH
   export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
   # Check that Tesseract is installed correctly:
   tesseract --version
   ```
   
# Configure build to install in your local directory
./autogen.sh
./configure --prefix=$HOME/.local
make -j$(nproc)
make install


4. **Index documents**
   ```bash
   python3 cli.py index --docdb-limit 10 --indico-limit 10 --start_idx_ddb 0 --start_idx_ind 0 
   ```
   start_idx_ddb/start_idx_ind: Which webpage to start at (-1 to not scrape from DocDB/Indico respectively)
   
   docdb-limit/indico-limit: How any webpages/categories to scrape respectively (-1 to scrape everything)

6. **Start the server**
   ```bash
   python3 cli.py serve
   ```

7. **Start server from local machine**
   ```bash
   ssh -L <port to open locally>:localhost:8000 <username to ssh into target machine> 
   ```
   ex:
   ```
   ssh -L 5000:localhost:8000 user@aurora.alcf.anl.gov
   ```
   Webpage launched locally on localhost:5000

Visit `http://localhost:<port>` to access the web interface as explained above.

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
- `DOC_LIMIT_DOCDB`: Number of DocDB documents to process (default: 50) Note: -1 for no limit (get all webpages)
- `DOC_LIMIT_INDICO`: Number of Indico documents to process (default: 50) Note: -1 for no limit (get all webpages)
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
dune_gpt/
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
│   │   └── fermilab_client.py   # Fermilab API client
│   ├── core/                # Core business logic
│   │   └── document_processor.py (single threaded with faiss)
│   │   └── document_processor_chroma.py (multi threaded with chroma)
│   │   └── document_processor_faiss.py (multi threaded with faiss)
│   ├── extractors/          # Document extractors
│   │   ├── base.py         # Base extractor class
│   │   ├── docdb_extractor.py
│   │   └── indico_extractor.py
│   │   ├── docdb_extractor_multithreaded.py
│   │   └── indico_extractor_multithreaded.py
│   ├── indexing/           # FAISS indexing
│   │   └── faiss_manager.py
│   │   └── chroma_manager.py
│   └── utils/              # Utilities
│       └── logger.py       # Logging setup
│
├── templates/              # HTML templates
│   └── index.html
├── static/                 # Static assets
│   └── images/
└── data/                   # Data storage
    └── faiss/              # FAISS index files
├── benchmarking/                     # Source code
│   ├── QuestionAnswer/                 # API clients
│   │   └── generateQA.py   # Uses Argo to generate test question answer pairs
│   │benchmarking_plot.py #Makes plots for different experiments
│   │test_models.sh #tests metrics against differet models   
│   │test_ks.sh #tests metrics against differet k values
│   │evaluation.py #Runs evaluation for Correctness, Source Retrieval, Latency
 
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

