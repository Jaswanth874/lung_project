# Lung Nodule Detection System

A comprehensive Flask-based web application for lung CT scan analysis with AI-powered nodule detection, user authentication, and clinical report generation using RAG (Retrieval-Augmented Generation).

## 🚀 Features

- **User Authentication**: Secure registration and login system
- **CT Scan Analysis**: Upload and analyze CT scans (.mhd) or images (PNG/JPG)
- **AI Detection**: Real-time nodule detection with bounding boxes and confidence scores
- **Clinical Reports**: RAG-enhanced report generation with PDF/TXT export
- **User Dashboard**: Track all uploaded scans and analysis history
- **Model Training**: Complete training pipeline with accuracy tracking and visualization

## 📋 Requirements

### Core Dependencies
- Python 3.9+
- Flask
- SQLAlchemy
- PyTorch (optional, for model inference)
- SimpleITK (for .mhd file support)
- Pillow
- NumPy
- Matplotlib (for training visualizations)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Gauri3112/lung-nodule-detection.git
cd lung-nodule-detection
```

2. **Install dependencies**:
```bash
pip install -r requirements-web.txt
pip install torch torchvision  # Optional: for model inference
pip install SimpleITK  # For CT scan file support
pip install matplotlib  # For training visualizations
```

3. **Set up environment variables** (optional):
Create a `.env` file:
```
FLASK_SECRET=your-secret-key-here
GOOGLE_API_KEY=your-google-gemini-key  # For RAG report generation
```

## 🏃 Running the Application

### Web Application (Flask)

```bash
python app.py
```

The app will be available at `http://localhost:5000`

### Training Models

```bash
python train_model.py
```

This will:
- Load LUNA16 dataset if available, or generate synthetic data
- Train an improved CNN model with batch normalization
- Track accuracy, precision, recall, and F1 scores
- Generate training history plots
- Save the trained model to `models/trained_model.pth`

## 📁 Project Structure

```
lung-nodule-detection/
├── app.py                      # Main Flask application
├── web_models.py               # SQLAlchemy database models
├── train_model.py              # Training script with accuracy tracking
├── requirements-web.txt        # Web app dependencies
├── requirements.txt            # ML dependencies
│
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── infer.py               # Model inference utilities
│   ├── data_loader.py         # CT scan loading
│   ├── preprocessing.py       # Image preprocessing
│   ├── train.py               # Training utilities with metrics
│   ├── ensemble.py            # Ensemble prediction
│   └── rag/                   # RAG module for report generation
│       ├── generator.py
│       ├── retriever.py
│       └── llm.py
│
├── templates/                  # Flask HTML templates
│   ├── layout.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── result.html
│   ├── dashboard.html
│   └── report.html
│
├── models/                     # Trained model files (gitignored)
├── outputs/                    # Generated predictions and reports (gitignored)
└── webapp.db                   # SQLite database (gitignored)
```

## 🔧 Usage

1. **Register/Login**: Create an account or log in
2. **Upload Scan**: Upload a CT scan (.mhd) or image file
3. **Analyze**: View detection results with confidence scores
4. **Generate Report**: Create and download clinical reports
5. **Dashboard**: View all your uploaded scans

## 📊 Model Architecture

The project includes two model architectures:

- **SimpleCNN**: Basic CNN for binary classification
- **ImprovedCNN**: Enhanced CNN with:
  - Batch normalization layers
  - Dropout for regularization
  - Adaptive pooling
  - 4 convolutional layers

## 📈 Training Features

- Accuracy tracking (training and validation)
- Precision, Recall, and F1 score metrics
- Learning rate scheduling
- Best model checkpointing
- Automatic plot generation
- Training history visualization

## 🔒 Security

- Password hashing with Werkzeug
- SQL injection protection via SQLAlchemy ORM
- Session management
- File upload validation

## 📝 Notes

- The database (`webapp.db`) is automatically created on first run
- Model files are stored in `models/` directory
- Generated outputs are saved in `outputs/` directory
- For production deployment, use a production WSGI server (Gunicorn/uWSGI)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Gauri3112**

## 🙏 Acknowledgments

- LUNA16 dataset for lung nodule detection
- Flask community
- PyTorch team

---

**Version**: 1.0  
**Last Updated**: February 2026  
**Status**: ✅ Active Development
