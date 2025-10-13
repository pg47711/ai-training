# ⚡ UV Setup Guide - Fastest Way to Get Started

## What is UV?

**UV** is a blazingly fast Python package manager (10-100x faster than pip!)

- ✅ **Fast**: Installs packages in seconds, not minutes
- ✅ **Simple**: One command to set up everything
- ✅ **Modern**: Uses pyproject.toml standard
- ✅ **Reliable**: Lock files for reproducible installs

---

## Quick Setup (< 1 minute)

### For Your Team:

```bash
# 1. Install UV (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone/download the RAG Demo project
cd "RAG Demo"

# 3. Run the setup script
./setup_with_uv.sh
```

**That's it!** Everything is installed and ready. ⚡

---

## What the Setup Script Does

```bash
./setup_with_uv.sh
```

**Automatically:**
1. ✅ Checks if UV is installed (installs if not)
2. ✅ Creates virtual environment (.venv)
3. ✅ Installs all dependencies (uses pyproject.toml)
4. ✅ Installs Jupyter for notebooks
5. ✅ Checks if Milvus is running
6. ✅ Checks if Ollama is installed
7. ✅ Shows next steps

**Time:** < 1 minute (UV is FAST!)

---

## Manual Setup (if you prefer)

### Step 1: Install UV

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Or with pip:**
```bash
pip install uv
```

---

### Step 2: Create Virtual Environment

```bash
cd "RAG Demo"
uv venv
```

Creates `.venv` folder.

---

### Step 3: Activate Virtual Environment

**macOS/Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```powershell
.venv\Scripts\activate
```

---

### Step 4: Install Dependencies

**Option A: From pyproject.toml (recommended)**
```bash
uv pip install -e .
```

**Option B: From requirements.txt**
```bash
uv pip install -r requirements.txt
```

**UV is 10-100x faster than regular pip!** ⚡

---

### Step 5: Install Jupyter (optional)

```bash
uv pip install jupyter jupyterlab
```

---

## Why UV Instead of Pip?

### Speed Comparison:

| Task | pip | UV | Speedup |
|------|-----|-----|---------|
| Install 50 packages | 2-3 min | 10-20 sec | **10x faster** |
| Create venv | 5 sec | 1 sec | **5x faster** |
| Resolve deps | 30 sec | 2 sec | **15x faster** |

### Other Benefits:

- **Reproducible**: Lock files ensure everyone gets same versions
- **Modern**: Uses pyproject.toml standard
- **Reliable**: Better dependency resolution
- **Compatible**: Works with existing requirements.txt

---

## Project Structure with UV

```
RAG Demo/
├── pyproject.toml           ← Project configuration (NEW!)
├── .python-version          ← Python version (NEW!)
├── setup_with_uv.sh        ← One-command setup (NEW!)
├── .venv/                   ← Virtual environment (created by UV)
├── requirements.txt         ← Still supported for compatibility
└── ... (notebooks, docs, etc.)
```

---

## Common UV Commands

### Install Project:
```bash
uv pip install -e .
```

### Add a Package:
```bash
uv pip install package-name
```

### Update All Packages:
```bash
uv pip install --upgrade -e .
```

### List Installed:
```bash
uv pip list
```

### Freeze Dependencies:
```bash
uv pip freeze > requirements-lock.txt
```

### Sync Environment:
```bash
# Ensures exact same packages as pyproject.toml
uv pip install -e . --force-reinstall
```

---

## For Your Team

### Share This Setup Process:

**Step 1:** Send them the project folder

**Step 2:** They run ONE command:
```bash
./setup_with_uv.sh
```

**Step 3:** Done! They can start working.

---

### Or Share pyproject.toml Approach:

```bash
# They just need:
pip install uv
uv venv
source .venv/bin/activate  # macOS/Linux
uv pip install -e .

# Done in < 1 minute!
```

---

## Optional Dependencies

### Install with extras:

```bash
# Install everything including dev tools
uv pip install -e ".[all]"

# Install just notebook support
uv pip install -e ".[notebook]"

# Install just dev tools
uv pip install -e ".[dev]"
```

**Defined in pyproject.toml:**
```toml
[project.optional-dependencies]
dev = ["jupyter", "jupyterlab", "ipywidgets"]
notebook = ["jupyter", "jupyterlab"]
all = ["rag-demo[dev,notebook]"]
```

---

## Environment Setup Comparison

### Old Way (pip):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # 2-3 minutes ⏱️
pip install jupyter jupyterlab
```
**Time:** 3-5 minutes

### New Way (UV):
```bash
./setup_with_uv.sh  # One command! ⚡
```
**Time:** 30-60 seconds

**Speedup:** 3-5x faster!

---

## Troubleshooting

### Issue: "uv: command not found"

**Solution:**
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Restart terminal
# Try again
```

---

### Issue: "Could not find a version that satisfies..."

**Solution:**
```bash
# UV resolves dependencies better, but if issues:
uv pip install -e . --no-deps
uv pip install -r requirements.txt
```

---

### Issue: Python version mismatch

**Solution:**
```bash
# UV can install Python versions too!
uv python install 3.13
uv venv --python 3.13
```

---

## CI/CD Integration

### GitHub Actions:

```yaml
name: Setup RAG Demo

jobs:
  setup:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install UV
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      
      - name: Setup environment
        run: |
          uv venv
          source .venv/bin/activate
          uv pip install -e .
      
      - name: Run tests
        run: |
          source .venv/bin/activate
          python -m pytest
```

---

## Migration from requirements.txt

### Both Supported!

You can still use:
```bash
pip install -r requirements.txt  # Old way (still works)
```

Or switch to:
```bash
uv pip install -e .  # New way (faster)
```

**We keep both** for maximum compatibility!

---

## Team Onboarding

### Send to new team members:

**Subject:** RAG Demo Setup - 1 Minute Install ⚡

**Body:**
```
Hi team!

To set up the RAG Demo project:

1. Install UV (one-time):
   curl -LsSf https://astral.sh/uv/install.sh | sh

2. Run setup:
   cd "RAG Demo"
   ./setup_with_uv.sh

3. Start Jupyter:
   jupyter lab

That's it! See you in the notebooks.

- Prabhu
```

---

## Advantages for Your Team

### Before (pip + requirements.txt):
- ❌ Slow installation (2-3 min)
- ❌ Manual venv creation
- ❌ Manual Jupyter install
- ❌ Version conflicts possible
- ❌ Different results on different machines

### After (UV + pyproject.toml):
- ✅ Fast installation (30-60 sec)
- ✅ Automatic venv creation
- ✅ Everything in one command
- ✅ Better dependency resolution
- ✅ Reproducible across machines
- ✅ Modern Python standard

---

## Summary

### What You Created:

1. **pyproject.toml** - Modern project config
   - All dependencies defined
   - Optional extras (dev, notebook)
   - Project metadata
   - Tool configurations

2. **setup_with_uv.sh** - One-command setup
   - Installs UV if needed
   - Creates venv
   - Installs all packages
   - Checks Milvus/Ollama
   - Shows next steps

3. **.python-version** - Python version pin
   - Ensures consistent Python version
   - Works with UV and pyenv

4. **UV_SETUP.md** - This guide
   - Complete UV documentation
   - Team onboarding instructions
   - Troubleshooting

---

## Quick Commands Reference

```bash
# Setup (one time)
./setup_with_uv.sh

# Activate venv
source .venv/bin/activate

# Install package
uv pip install package-name

# Update dependencies
uv pip install -e . --upgrade

# Start Jupyter
jupyter lab

# Run agentic RAG
python agentic_rag.py

# Visualize
python visualize_workflows.py
```

---

**🚀 Your team can now set up the entire project in < 1 minute!**

**Share:** `setup_with_uv.sh` and `UV_SETUP.md` with your team.
