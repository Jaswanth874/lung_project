# Lung Nodule AI Web Application

A complete Flask-based web application for lung CT scan analysis with user authentication, detection reporting, and clinical insights via RAG (Retrieval-Augmented Generation).

## Features

✅ **User Authentication**
- Registration & login with hashed passwords
- Secure session management
- User dashboard with upload history

✅ **CT Scan Analysis**
- Upload CT scans (.mhd) or images (PNG/JPG)
- Real-time nodule detection with bounding boxes
- Confidence scoring (0–1 scale)
- Visual analysis output with annotated lesions

✅ **Reporting**
- Automatic clinical report generation using RAG
- Download PDF/TXT reports
- Detection region insights
- Database-backed report storage

✅ **Database**
- SQLAlchemy ORM with SQLite
- User records
- CT scan metadata
- Detection results
- Clinical reports

## Installation

1. **Clone/Open the project:**
   ```bash
   cd "C:\Users\91938\OneDrive\Desktop\lung project\lung project"
   ```

2. **Create virtual environment (if not already done):**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements-web.txt
   pip install google-generativeai python-dotenv
   ```

4. **Run migrations (optional - DB is auto-created):**
   ```bash
   python web_models.py
   ```

## Running the App

```bash
python app.py
```

Then open **http://localhost:8501** in your browser.

### Default Flow

1. **Register** (top-right) → Create account with email + password
2. **Login** → Access the analysis page
3. **Upload** → Choose a CT scan (.mhd) or image (PNG/JPG)
4. **Analyze** → View detected lesions and confidence score
5. **Generate Report** → Download clinical report (RAG-enhanced)
6. **Dashboard** → View all your uploaded scans

## Project Structure

```
lung project/
├── app.py                      # Main Flask app (routes, auth, analysis)
├── web_models.py              # SQLAlchemy models (User, CTScan, etc.)
├── requirements-web.txt       # Web app dependencies
│
├── templates/
│   ├── layout.html            # Base template
│   ├── index.html             # Home / analysis page
│   ├── login.html             # Login form
│   ├── register.html          # Registration form
│   ├── result.html            # Analysis result + report button
│   └── dashboard.html         # User's scan history
│
├── src/
│   ├── infer.py              # Inference utilities (predict, detect_boxes, NMS)
│   ├── data_loader.py        # CT scan loading
│   ├── preprocessing.py      # Image normalization
│   └── rag/
│       ├── generator.py      # RAG report generation
│       ├── retriever.py      # Knowledge retrieval
│       └── llm.py           # Google Gemini LLM integration
│
├── models/
│   ├── baseline_model.pth    # Model weights
│   └── retinanet_best.pth    # RetinaNet model
│
├── outputs/
│   ├── predictions/          # Generated analysis images
│   └── reports/              # Generated clinical reports
│
└── webapp.db                 # SQLite database (auto-created)
```

## Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Home / analysis page (requires login) |
| `/register` | GET, POST | User registration |
| `/login` | GET, POST | User login |
| `/logout` | GET | Logout |
| `/analyze` | POST | Analyze CT scan / image |
| `/generate_report` | POST | Generate & download clinical report |
| `/dashboard` | GET | View user's scan history |
| `/outputs/predictions/<fname>` | GET | Serve analysis images |

## Environment Variables

Create a `.env` file in the project root:
```
FLASK_SECRET=your-secret-key-here
GOOGLE_API_KEY=your-google-gemini-key
```

The app will use defaults if not set, but for production RAG, set `GOOGLE_API_KEY`.

## Database

SQLite database is automatically created at `webapp.db` on first run.

**Tables:**
- `users` — User accounts
- `ct_scans` — Uploaded CT scans
- `detection_results` — Bounding boxes & confidence scores
- `clinical_reports` — Generated reports

## Notes

- **Development server** (Flask debug mode) is used. For production, use Gunicorn/uWSGI.
- **Image uploads** are stored in `outputs/predictions/`
- **CT scans (.mhd)** are extracted to temp files, then deleted after analysis
- **RAG** requires a valid Google Gemini API key for full report generation

## Troubleshooting

**Port already in use:**
```bash
# Change port in app.py last line:
app.run(host='0.0.0.0', port=8502, debug=True)  # Use 8502 instead
```

**Missing modules:**
```bash
pip install -r requirements-web.txt
pip install google-generativeai python-dotenv
```

**DB locked:**
```bash
# Delete and recreate:
rm webapp.db
python web_models.py
```

## Future Enhancements

- [ ] Async inference with Celery
- [ ] PDF report export with matplotlib plots
- [ ] Multi-user batch processing
- [ ] API endpoints (REST/GraphQL)
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP)

---

**Version:** 1.0  
**Last Updated:** Feb 12, 2026  
**Status:** ✅ Running on http://localhost:8501
