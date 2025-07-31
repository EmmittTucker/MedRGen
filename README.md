# Medical Reasoning Dataset Generator (MedRGen)

A professional synthetic medical reasoning dataset generator that creates high-quality medical cases through simulated doctor-patient interactions. Built for commercial licensing and production use.

## 🎯 Project Overview

MedRGen generates comprehensive synthetic medical datasets by simulating realistic doctor-patient interactions and diagnostic processes. The system produces dual-quality datasets:

- **Raw Dataset**: Unedited generated medical cases for research and development
- **Evaluated Dataset**: LLM-reviewed and refined cases for production use

## ✨ Key Features

- 🏥 **Multi-Specialty Support**: Internal medicine, cardiology, emergency medicine, family medicine
- 👥 **Realistic Demographics**: Diverse patient profiles with appropriate medical histories
- 💬 **Natural Conversations**: Multi-turn doctor-patient dialogues following clinical patterns
- 🧠 **Evidence-Based Reasoning**: Medical expert reasoning using current clinical guidelines
- 📊 **Quality Assurance**: Dual dataset generation with LLM evaluation
- 🔧 **Professional Architecture**: SOLID principles, comprehensive logging, modular design
- 📈 **Commercial Ready**: Designed for licensing and production deployment

## 📋 Requirements

- Python 3.9+
- OpenAI API access (GPT-4 and O1-preview models)
- Windows PowerShell (recommended) or Anaconda
- 4GB+ RAM recommended for batch processing

## 🚀 Quick Start

### 1. Environment Setup

```powershell
# Using Anaconda (recommended)
conda create -n medical-dataset python=3.9
conda activate medical-dataset

# Or using Python venv
python -m venv medical-dataset-env
medical-dataset-env\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure API Keys

Copy `env.example` to `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MEDICAL_MODEL=gpt-4-turbo
OPENAI_REASONING_MODEL=o1-preview
```

### 4. Generate Your First Medical Case

```powershell
python run_generator.py --generate-case --specialty cardiology --complexity moderate
```

## 📖 Usage Examples

### Single Case Generation

```powershell
# Generate a cardiology case
python run_generator.py --generate-case --specialty cardiology --complexity moderate

# Generate with specific symptom theme
python run_generator.py --generate-case --theme cardiovascular --complexity complex

# Generate case with evaluation (creates both raw and evaluated versions)
python run_generator.py --generate-case --specialty cardiology --evaluate
```

### Batch Generation

```powershell
# Generate 10 cases with varied specialties
python run_generator.py --generate-batch --count 10 --complexity moderate

# Generate themed batch
python run_generator.py --generate-batch --count 50 --theme respiratory --specialty internal_medicine

# Generate batch with evaluation (dual datasets)
python run_generator.py --generate-batch --count 25 --evaluate
```

### Dual Dataset Generation

```powershell
# Generate evaluated datasets with quality improvements
python run_generator.py --generate-batch --count 100 --evaluate

# Raw datasets saved to: data/output/raw_dataset/
# Evaluated datasets saved to: data/output/edited_dataset/
```

## 🏗️ Project Structure

```
MedRGen/
├── src/
│   ├── config/prompts/          # YAML configuration files
│   ├── core/                    # Generation engines
│   │   ├── medical_expert_generator.py
│   │   ├── patient_generator.py
│   │   └── doctor_patient_conversation_generator.py
│   ├── models/                  # Pydantic data models
│   │   ├── doctor.py
│   │   ├── patient.py
│   │   └── medical_case.py
│   ├── utils/                   # Utilities
│   │   └── logging_setup.py
│   └── main.py                  # CLI interface
├── data/
│   ├── output/raw_dataset/      # Generated cases (raw)
│   └── output/edited_dataset/   # Evaluated cases
├── tests/                       # Test suite
├── logs/                        # Application logs
└── docs/                        # Documentation
```

## 🎭 Generated Case Example

Each generated case includes:

- **Patient Profile**: Demographics, medical history, medications, allergies
- **Chief Complaint**: Primary reason for visit
- **Symptom History**: Detailed chronological progression  
- **Physical Exam**: Vital signs and examination findings
- **Diagnostic Reasoning**: Step-by-step medical reasoning
- **Differential Diagnoses**: Multiple diagnostic possibilities with rationales
- **Treatment Plan**: Immediate, short-term, and long-term management
- **Conversation Transcript**: Complete doctor-patient dialogue
- **Patient Education**: Layperson explanation of condition

## 🧪 Testing

Run the comprehensive test suite:

```powershell
# Run all tests
python tests/run_all_tests.py

# Check system requirements
python tests/run_all_tests.py --check-requirements

# Run with different verbosity
python tests/run_all_tests.py --verbosity 2
```

## 🔧 Configuration

### Medical Specialties
Configure in `src/config/prompts/medical_specialties.yaml`:
- Internal Medicine
- Cardiology  
- Emergency Medicine
- Family Medicine

### Symptom Themes
Configure in `src/config/prompts/symptom_themes.yaml`:
- Cardiovascular
- Respiratory
- Gastrointestinal
- Neurological
- Musculoskeletal

### Demographics
Configure in `src/config/prompts/demographic_variations.yaml`:
- Age groups (pediatric, adult, geriatric)
- Sex variations and considerations
- Socioeconomic factors
- Cultural backgrounds

## 📊 Quality Assurance

The system includes multiple quality assurance layers:

- **Medical Accuracy Validation**: Clinical guideline compliance
- **Conversation Naturalness**: Human-like dialogue patterns
- **Diagnostic Consistency**: Logical reasoning flow
- **Safety Considerations**: Critical finding identification
- **Demographic Diversity**: Appropriate variation in presentations

## 💼 Commercial Features

- **Dual Dataset Generation**: Raw and evaluated versions
- **Scalable Architecture**: Batch processing with rate limiting
- **Professional Logging**: Structured logging for monitoring
- **Quality Metrics**: Comprehensive evaluation scoring
- **Export Formats**: JSON with extensible schema
- **Documentation**: Commercial-grade documentation

## 📈 Roadmap

- [x] **Dual Dataset Generation** (Raw + Evaluated with o3-mini)
- [x] **Comprehensive Field Population** (No null fields in evaluated datasets)
- [x] **Quality Scoring and Metrics**
- [ ] Advanced medical validation with knowledge bases
- [ ] Additional medical specialties (oncology, psychiatry, pediatrics)
- [ ] Multi-language support
- [ ] Integration with external medical knowledge bases
- [ ] Real-time quality scoring with live feedback
- [ ] Custom prompt templates for specialized use cases
- [ ] Advanced batch processing optimization

## 🤝 Contributing

This is a commercial project. For enterprise licensing, custom development, or collaboration opportunities, please contact the development team.

## 📄 License

Commercial License - See licensing documentation for terms and conditions.

## 🔒 Privacy & Security

- No PHI (Protected Health Information) is used
- All generated cases are synthetic
- API keys and sensitive data are properly secured
- Comprehensive .gitignore for data protection

## 📞 Support

For technical support, feature requests, or commercial inquiries:
- Documentation: `docs/` directory
- Logs: Check `logs/` for detailed execution information
- Issues: Review test output and logs for troubleshooting

---

**Built with ❤️ for advancing medical AI research and education**

