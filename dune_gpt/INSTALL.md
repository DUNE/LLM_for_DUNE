# DUNE-GPT Installation Guide

This guide provides step-by-step instructions for installing and running DUNE-GPT.

## 📋 Prerequisites

- **Python 3.10+** (check with `python3 --version`)
- **Git** (for cloning the repository)
- **DUNE DocDB credentials**
- **Argo API credentials**
- **Fermilab OAuth2 credentials** (Client ID & Secret)
- **Docker** (optional, for containerized deployment)

---

## 🚀 First-Time Installation

### Step 1: Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd DUNEGPT_polaris_new

# Create a Python virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### Step 2: Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

### Step 3: Configure Environment Variables

```bash
# Copy the environment template
cp env.example .env

# Edit the .env file with your credentials
nano .env  # or use your preferred editor
```

**Required settings in `.env`:**
```bash
# Argo API (Required)
ARGO_API_USERNAME=your_argo_username
ARGO_API_KEY=your_argo_api_key

# DUNE DocDB (Required)
DUNE_DOCDB_USERNAME=your_docdb_username
DUNE_DOCDB_PASSWORD=your_docdb_password

# Fermilab Authentication (Required if enabled)
ENABLE_AUTHENTICATION=true
FERMILAB_CLIENT_ID=your_fermilab_client_id
FERMILAB_CLIENT_SECRET=your_fermilab_client_secret
FERMILAB_REDIRECT_URI=http://127.0.0.1:8000/auth

# Optional
DUNE_INDICO_ACCESS_KEY=your_indico_access_key
DOC_LIMIT_DOCDB=50
DOC_LIMIT_INDICO=50
```

### Step 4: Verify Configuration

```bash
# Check system health and configuration
python cli.py health
```

You should see:
- ✅ Configuration: OK
- ✅ Fermilab Authentication: Configured (if enabled)

### Step 5: Index Documents

```bash
# Extract and index documents (first time only)
python cli.py index --docdb-limit 50 --indico-limit 50
```

This will:
- Extract documents from DUNE DocDB
- Extract documents from Indico
- Generate embeddings
- Build FAISS index
- Save everything to `data/faiss/`

**Expected output:**
```
INDEXING RESULTS
==================================================
DocDB documents processed: X
Indico documents processed: Y
Total new documents added: Z
```

### Step 6: Start the Application

```bash
# Start the web server
python cli.py serve
```

Visit `http://127.0.0.1:8000` in your browser.

### Step 7: Test Authentication (if enabled)

1. Click "Login with Fermilab"
2. Authenticate with your Fermilab credentials
3. Return to DUNE-GPT
4. Ask a question about DUNE

---

## 🔄 Subsequent Runs

### Quick Start (Already Configured)

```bash
# Navigate to project directory
cd /path/to/DUNEGPT_polaris_new

# Activate virtual environment (if using)
source .venv/bin/activate

# Start the application
python cli.py serve
```

### Update Documents (Periodic)

```bash
# Add new documents to existing index
python cli.py index --docdb-limit 100 --indico-limit 100

# Check index statistics
python cli.py stats
```

### Health Check

```bash
# Verify everything is working
python cli.py health
```

---

## 🐳 Docker Deployment

### First-Time Docker Setup

```bash
# Ensure .env file is configured
cp env.example .env
# Edit .env with your credentials

# Build and start with Docker Compose
docker-compose up --build

# In another terminal, index documents
docker-compose exec dune-gpt python cli.py index
```

### Subsequent Docker Runs

```bash
# Start existing containers
docker-compose up

# Stop containers
docker-compose down
```

---

## 🔧 Common Commands

### CLI Commands Reference

```bash
# Document management
python cli.py index                    # Index new documents
python cli.py stats                    # Show index statistics
python cli.py query "your question"    # Query from command line

# System management
python cli.py health                   # System health check
python cli.py serve                    # Start web server

# Help
python cli.py --help                   # Show all commands
python cli.py index --help             # Show command options
```

### API Endpoints

```bash
# Health check
curl http://127.0.0.1:8000/api/health

# Search query
curl "http://127.0.0.1:8000/api/search?q=DUNE%20detector&top_k=3"

# Index statistics
curl http://127.0.0.1:8000/api/stats
```

---

## 🔒 Authentication Options

### Enable Authentication
```bash
# In .env file
ENABLE_AUTHENTICATION=true
FERMILAB_CLIENT_ID=your_client_id
FERMILAB_CLIENT_SECRET=your_client_secret
```

### Disable Authentication (Open Access)
```bash
# In .env file
ENABLE_AUTHENTICATION=false
```

---

## 🛠️ Troubleshooting

### Environment Issues
```bash
# Check Python version
python3 --version  # Should be 3.10+

# Check installed packages
pip list | grep -E "(fastapi|torch|faiss)"

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Index Issues
```bash
# Check index status
python cli.py stats

# Rebuild index from scratch
rm -rf data/faiss/*
python cli.py index
```

### Authentication Issues
```bash
# Verify credentials
python cli.py health

# Check logs for OAuth2 errors
python cli.py serve  # Watch console output
```

### Docker Issues
```bash
# Rebuild containers
docker-compose down
docker-compose up --build

# Check container logs
docker-compose logs dune-gpt
```

---

## 📁 File Structure

```
DUNEGPT_polaris_new/
├── cli.py              # Command-line interface
├── main.py             # Web application
├── config.py           # Configuration
├── .env                # Your credentials (create from env.example)
├── data/faiss/         # FAISS index files
├── src/                # Source code modules
├── templates/          # Web UI templates
├── static/             # Web assets
└── requirements.txt    # Dependencies
```

---

## 🔄 Update Workflow

### Regular Updates
1. Pull latest code: `git pull`
2. Update dependencies: `pip install -r requirements.txt`
3. Update documents: `python cli.py index`
4. Restart server: `python cli.py serve`

### Add More Documents
```bash
# Increase document limits
python cli.py index --docdb-limit 200 --indico-limit 200
```

---

## 💡 Tips

- **Virtual Environment**: Always use a virtual environment to avoid conflicts
- **Document Limits**: Start with small limits (50) for testing, increase for production
- **Index Persistence**: The FAISS index is saved in `data/faiss/` and persists between runs
- **Memory Usage**: Higher document counts require more RAM
- **Authentication**: Can be toggled on/off without rebuilding the index

---

## 📞 Support

- Check logs with `python cli.py serve` for debugging
- Use `python cli.py health` to verify system status
- Index statistics: `python cli.py stats`
- For authentication issues, verify your Fermilab OAuth2 app configuration

---

**🎉 You're ready to use DUNE-GPT!** 