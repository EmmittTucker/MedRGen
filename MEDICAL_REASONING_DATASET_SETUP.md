# Medical Reasoning Dataset Generator - Setup Guide

## Project Overview

This project generates high-quality medical reasoning datasets by simulating doctor-patient interactions and medical diagnostic processes. Built as an extension of the FAGen_InvestmentAdvisor framework, it leverages commercial AI models to create comprehensive medical cases with dual-quality datasets (unedited and LLM-edited versions).

## Project Objectives

1. **Generate Synthetic Medical Cases**: Create diverse, realistic medical scenarios with complete diagnostic workflows
2. **Multi-turn Conversation Simulation**: Simulate natural doctor-patient interactions for training conversational AI
3. **Quality Assurance**: Provide both raw and LLM-evaluated datasets for different use cases
4. **Commercial Viability**: Produce publication-ready datasets suitable for commercial licensing

## Dataset Schema

```json
{
  "case_id": "unique_identifier",
  "patient_profile": {
    "age": 45,
    "sex": "M/F/Other",
    "occupation": "string",
    "medical_history": ["condition1", "condition2"],
    "medications": ["med1", "med2"],
    "allergies": ["allergy1", "allergy2"],
    "family_history": ["condition1", "condition2"],
    "social_history": {
      "smoking": "status",
      "alcohol": "status",
      "exercise": "frequency"
    }
  },
  "chief_complaint": "Primary reason for visit",
  "symptom_history": "Detailed chronological symptom progression",
  "physical_exam": {
    "vital_signs": {...},
    "general_appearance": "...",
    "system_specific_findings": {...}
  },
  "labs": {
    "blood_work": {...},
    "imaging": {...},
    "specialized_tests": {...}
  },
  "differential_diagnoses": [
    {
      "diagnosis": "Primary suspected condition",
      "rationale": "Evidence-based reasoning",
      "probability": "likelihood_score"
    }
  ],
  "reasoning_steps": "Step-by-step diagnostic thought process",
  "final_diagnosis": "Confirmed diagnosis with ICD-10 code",
  "treatment_plan": {
    "immediate": ["action1", "action2"],
    "short_term": ["plan1", "plan2"],
    "long_term": ["goal1", "goal2"],
    "monitoring": ["parameter1", "parameter2"]
  },
  "patient_explanation": "Layperson explanation of condition and treatment",
  "conversation_transcript": [
    {
      "speaker": "doctor/patient",
      "content": "utterance",
      "timestamp": "relative_time",
      "intent": "question/answer/clarification"
    }
  ],
  "metadata": {
    "specialty": "internal_medicine/cardiology/etc",
    "complexity": "simple/moderate/complex",
    "theme": "symptom_category",
    "generation_timestamp": "datetime",
    "llm_edited": true/false
  }
}
```

## Project Structure

```
MedicalDatasetGenerator/
├── src/
│   ├── config/
│   │   └── prompts/
│   │       ├── medical_specialties.yaml
│   │       ├── symptom_themes.yaml
│   │       ├── demographic_variations.yaml
│   │       ├── reasoning_constraints.yaml
│   │       └── conversation_flows.yaml
│   ├── core/
│   │   ├── __init__.py
│   │   ├── medical_expert_generator.py
│   │   ├── doctor_patient_conversation_generator.py
│   │   ├── patient_generator.py
│   │   ├── medical_context.py
│   │   └── dataset_evaluator.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── doctor.py
│   │   ├── patient.py
│   │   └── medical_case.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_export.py
│   │   ├── logging_setup.py
│   │   └── medical_validation.py
│   └── main.py
├── data/
│   ├── output/
│   │   ├── raw_dataset/
│   │   └── edited_dataset/
│   └── templates/
├── tests/
│   ├── test_medical_models.py
│   ├── test_conversation_generation.py
│   ├── test_dataset_quality.py
│   └── run_all_tests.py
├── docs/
├── logs/
├── requirements.txt
├── env.example
└── README.md
```

## Environment Setup

### 1. Prerequisites

- Python 3.9+
- Anaconda or Miniconda (recommended)
- Windows PowerShell (for Windows users)
- OpenAI API access with GPT-4 and reasoning model capabilities

### 2. Virtual Environment Setup

```powershell
# Using Anaconda (recommended)
conda create -n medical-dataset python=3.9
conda activate medical-dataset

# Or using Python venv
python -m venv medical-dataset-env
medical-dataset-env\Scripts\Activate.ps1
```

### 3. Dependencies Installation

```powershell
pip install -r requirements.txt
```

### Required Dependencies:
```txt
openai>=1.0.0
pydantic>=2.0.0
pyyaml>=6.0
pandas>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0
python-dotenv>=1.0.0
jsonschema>=4.17.0
pytest>=7.0.0
pytest-cov>=4.0.0
logging>=0.4.9.6
```

### 4. API Configuration

Create `.env` file in project root:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MEDICAL_MODEL=gpt-4-turbo
OPENAI_REASONING_MODEL=o1-preview

# Generation Settings
MAX_CASES_PER_BATCH=10
CONCURRENT_REQUESTS=3
RATE_LIMIT_DELAY=1

# Dataset Configuration
OUTPUT_FORMAT=json
INCLUDE_METADATA=true
VALIDATE_MEDICAL_ACCURACY=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/medical_dataset_generation.log
```

## Core Components Implementation

### 1. Medical Expert Generator (`medical_expert_generator.py`)
- **Purpose**: Generate medical experts with different specialties and reasoning styles
- **Key Features**: 
  - Specialty-specific knowledge bases
  - Reasoning pattern variations
  - Evidence-based decision making
- **Mapped from**: `advisor_generator.py`

### 2. Doctor-Patient Conversation Generator (`doctor_patient_conversation_generator.py`)
- **Purpose**: Create realistic multi-turn medical consultations
- **Key Features**:
  - Natural conversation flow
  - Medical interview techniques
  - Patient communication styles
- **Enhanced from**: `conversation_generator.py`

### 3. Patient Generator (`patient_generator.py`)
- **Purpose**: Create diverse patient profiles and presentations
- **Key Features**:
  - Demographic variation
  - Medical history complexity
  - Symptom presentation patterns
- **Mapped from**: `investor_generator.py`

### 4. Medical Context (`medical_context.py`)
- **Purpose**: Provide medical knowledge context and constraints
- **Key Features**:
  - Disease prevalence data
  - Symptom-diagnosis correlations
  - Treatment guideline adherence
- **Mapped from**: `market_context.py`

### 5. Dataset Evaluator (`dataset_evaluator.py`)
- **Purpose**: Quality assurance using reasoning models
- **Key Features**:
  - Medical accuracy validation
  - Reasoning coherence assessment
  - Clinical guideline compliance
- **New Component**: Unique to medical project

## Configuration Files

### 1. Medical Specialties (`medical_specialties.yaml`)
```yaml
specialties:
  internal_medicine:
    focus_areas: ["general_medicine", "chronic_disease", "preventive_care"]
    complexity_levels: ["routine", "moderate", "complex"]
    common_presentations: ["fatigue", "chest_pain", "shortness_of_breath"]
  
  cardiology:
    focus_areas: ["heart_disease", "arrhythmias", "interventional"]
    complexity_levels: ["stable", "acute", "critical"]
    common_presentations: ["chest_pain", "palpitations", "syncope"]
```

### 2. Symptom Themes (`symptom_themes.yaml`)
```yaml
themes:
  cardiovascular:
    primary_symptoms: ["chest_pain", "shortness_of_breath", "palpitations"]
    associated_symptoms: ["fatigue", "dizziness", "edema"]
    demographics: ["middle_aged", "elderly", "risk_factors"]
  
  respiratory:
    primary_symptoms: ["cough", "dyspnea", "chest_tightness"]
    associated_symptoms: ["fever", "sputum", "wheezing"]
    demographics: ["all_ages", "smoking_history", "environmental"]
```

### 3. Demographic Variations (`demographic_variations.yaml`)
```yaml
age_groups:
  pediatric: {min: 0, max: 17, special_considerations: ["growth", "development"]}
  adult: {min: 18, max: 64, special_considerations: ["occupational", "reproductive"]}
  geriatric: {min: 65, max: 100, special_considerations: ["polypharmacy", "frailty"]}

sex_variations:
  male: {conditions: ["prostate", "testosterone"], presentations: ["atypical_mi"]}
  female: {conditions: ["pregnancy", "menopause"], presentations: ["autoimmune"]}
```

## Implementation Workflow

### Phase 1: Core Infrastructure
1. Set up project structure and environment
2. Implement base models (Doctor, Patient, MedicalCase)
3. Create configuration management system
4. Establish logging and error handling

### Phase 2: Generation Engine
1. Implement MedicalExpertGenerator
2. Develop PatientGenerator with demographic variations
3. Create MedicalContext for knowledge constraints
4. Build conversation flow engine

### Phase 3: Quality Assurance
1. Implement DatasetEvaluator with reasoning model
2. Create medical validation utilities
3. Develop dual-output system (raw/edited)
4. Establish quality metrics and scoring

### Phase 4: Testing and Validation
1. Create comprehensive test suite
2. Validate medical accuracy with domain experts
3. Performance testing and optimization
4. Dataset quality assessment

### Phase 5: Production and Documentation
1. Automated generation pipeline
2. Dataset export and packaging
3. Documentation and usage guides
4. Commercial licensing preparation

## Dataset Generation Process

### 1. Single Case Generation
```bash
python src/main.py --generate-case --specialty cardiology --complexity moderate --count 1
```

### 2. Batch Generation
```bash
python src/main.py --generate-batch --theme cardiovascular --count 100 --output data/output/raw_dataset/
```

### 3. Quality Evaluation
```bash
python src/main.py --evaluate-dataset --input data/output/raw_dataset/ --output data/output/edited_dataset/
```

### 4. Full Pipeline
```bash
python src/main.py --full-pipeline --theme respiratory --count 500 --evaluate
```

## Quality Assurance Framework

### Medical Accuracy Validation
- **Clinical Guidelines Compliance**: Adherence to established medical protocols
- **Diagnostic Reasoning**: Logical progression from symptoms to diagnosis
- **Treatment Appropriateness**: Evidence-based treatment recommendations
- **Safety Considerations**: Identification of critical findings and contraindications

### Technical Quality Metrics
- **Conversation Naturalness**: Human-like dialogue patterns
- **Information Completeness**: All required schema fields populated
- **Consistency**: Internal logical consistency across case elements
- **Diversity**: Adequate variation in presentations and demographics

## Testing Strategy

### Unit Tests
- Model validation and serialization
- Generation logic correctness
- Configuration parsing
- Utility function reliability

### Integration Tests
- End-to-end case generation
- API integration functionality
- Data export pipeline
- Quality evaluation workflow

### Medical Validation Tests
- Clinical accuracy assessment
- Specialty-specific content validation
- Treatment guideline compliance
- Safety consideration coverage

## Commercial Considerations

### Dataset Packaging
- **Standard Dataset**: Raw, unedited cases for research and development
- **Premium Dataset**: LLM-evaluated and refined cases for production use
- **Custom Datasets**: Specialty-specific or constraint-based generation

### Licensing Structure
- **Academic License**: Reduced pricing for educational institutions
- **Commercial License**: Full commercial usage rights
- **Enterprise License**: Custom generation and ongoing support

### Quality Assurance Documentation
- Medical expert validation reports
- Statistical diversity analysis
- Bias assessment and mitigation
- Compliance with healthcare AI standards

## Maintenance and Updates

### Regular Updates
- Medical guideline updates
- New specialty incorporation
- Model performance improvements
- Dataset quality enhancements

### Monitoring
- Generation success rates
- Medical accuracy metrics
- User feedback integration
- Performance optimization

## Getting Started

1. **Clone the repository structure**
2. **Set up the environment** using the provided instructions
3. **Configure API keys** and settings in `.env`
4. **Run initial tests** to verify setup
5. **Generate sample cases** to validate functionality
6. **Scale to production** dataset generation

## Support and Documentation

- **Technical Documentation**: Detailed API references and implementation guides
- **Medical Guidelines**: Clinical validation criteria and accuracy standards
- **Usage Examples**: Sample implementations and use cases
- **Troubleshooting**: Common issues and resolution procedures

---

**Note**: This setup guide provides the complete framework for transitioning from the financial advisor dataset generator to a medical reasoning dataset generator. Follow the implementation phases sequentially for optimal results and maintain the same professional standards and modular architecture as the original project.