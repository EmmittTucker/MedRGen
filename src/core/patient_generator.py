"""
Patient Generator for synthetic medical case creation.

This module generates diverse patient profiles with realistic demographics,
medical histories, social backgrounds, and symptom presentations based on
epidemiological data and demographic variations.
"""

import random
import uuid
import yaml
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta

from models import (
    Patient, PatientProfile, SocialHistory, VitalSigns, 
    PhysicalExam, LabResults
)
from utils.logging_setup import get_logger


class PatientGenerator:
    """
    Generates diverse synthetic patients for medical case creation.
    
    This class creates realistic patient profiles with appropriate demographic
    variations, medical histories, and symptom presentations based on 
    epidemiological data and clinical guidelines.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Patient Generator.
        
        Args:
            config_path: Path to configuration files
        """
        self.logger = get_logger(f"{__name__}.PatientGenerator")
        
        # Load configuration
        self.config_path = Path(config_path) if config_path else Path(__file__).parent.parent / "config" / "prompts"
        self.demographics_config = self._load_yaml_config("demographic_variations.yaml")
        self.symptoms_config = self._load_yaml_config("symptom_themes.yaml")
        
        # Common medical conditions by age group
        self.age_based_conditions = self._initialize_age_based_conditions()
        
        # Common medications by condition
        self.condition_medications = self._initialize_condition_medications()
        
        self.logger.info("PatientGenerator initialized successfully")
    
    def _load_yaml_config(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        config_file = self.config_path / filename
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_file}")
            return {}
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML config {filename}: {e}")
            return {}
    
    def _initialize_age_based_conditions(self) -> Dict[str, List[str]]:
        """Initialize common medical conditions by age group."""
        return {
            "pediatric": [
                "Asthma", "Allergic rhinitis", "Attention deficit disorder", 
                "Recurrent ear infections", "Eczema"
            ],
            "young_adult": [
                "Anxiety disorder", "Depression", "Migraine headaches", 
                "Gastroesophageal reflux", "Acne"
            ],
            "middle_aged": [
                "Hypertension", "Type 2 diabetes", "Hyperlipidemia", 
                "Osteoarthritis", "Sleep apnea", "Depression"
            ],
            "geriatric": [
                "Hypertension", "Type 2 diabetes", "Coronary artery disease",
                "Osteoarthritis", "Chronic kidney disease", "Dementia",
                "Atrial fibrillation", "Heart failure"
            ]
        }
    
    def _initialize_condition_medications(self) -> Dict[str, List[str]]:
        """Initialize common medications by medical condition."""
        return {
            "Hypertension": [
                "Lisinopril 10mg daily", "Amlodipine 5mg daily", 
                "Metoprolol 50mg BID", "Hydrochlorothiazide 25mg daily"
            ],
            "Type 2 diabetes": [
                "Metformin 500mg BID", "Glipizide 5mg daily",
                "Insulin glargine 20 units bedtime", "Sitagliptin 100mg daily"
            ],
            "Hyperlipidemia": [
                "Atorvastatin 20mg daily", "Simvastatin 40mg daily",
                "Rosuvastatin 10mg daily"
            ],
            "Depression": [
                "Sertraline 50mg daily", "Fluoxetine 20mg daily",
                "Escitalopram 10mg daily"
            ],
            "Asthma": [
                "Albuterol inhaler PRN", "Budesonide/formoterol inhaler BID",
                "Montelukast 10mg daily"
            ],
            "Osteoarthritis": [
                "Ibuprofen 400mg TID PRN", "Acetaminophen 650mg QID PRN",
                "Meloxicam 15mg daily"
            ]
        }
    
    def generate_patient(
        self,
        age_group: Optional[str] = None,
        sex: Optional[str] = None,
        symptom_theme: Optional[str] = None,
        complexity: str = "moderate"
    ) -> Patient:
        """
        Generate a complete patient profile.
        
        Args:
            age_group: Specific age group (pediatric, young_adult, middle_aged, geriatric)
            sex: Patient sex (M, F, Other)
            symptom_theme: Theme for symptoms (cardiovascular, respiratory, etc.)
            complexity: Case complexity (simple, moderate, complex)
        
        Returns:
            Generated Patient instance
        """
        self.logger.info(f"Generating patient - Age group: {age_group}, Sex: {sex}, Theme: {symptom_theme}")
        
        # Generate demographics
        demographics = self._generate_demographics(age_group, sex)
        
        # Generate medical history based on demographics and complexity
        medical_history = self._generate_medical_history(demographics, complexity)
        
        # Generate medications based on medical history
        medications = self._generate_medications(medical_history)
        
        # Generate social history
        social_history = self._generate_social_history(demographics)
        
        # Create patient profile
        patient_profile = PatientProfile(
            age=demographics["age"],
            sex=demographics["sex"],
            occupation=demographics["occupation"],
            medical_history=medical_history,
            medications=medications,
            allergies=demographics["allergies"],
            family_history=demographics["family_history"],
            social_history=social_history
        )
        
        # Generate chief complaint and symptom history
        chief_complaint, symptom_history, current_symptoms = self._generate_symptoms(
            patient_profile, symptom_theme, complexity
        )
        
        # Create patient
        patient = Patient(
            patient_id=f"PAT_{uuid.uuid4().hex[:8].upper()}",
            profile=patient_profile,
            chief_complaint=chief_complaint,
            symptom_history=symptom_history,
            current_symptoms=current_symptoms,
            symptom_severity=self._generate_symptom_severity(current_symptoms),
            pain_scale=random.randint(3, 8) if "pain" in chief_complaint.lower() else None
        )
        
        self.logger.info(f"Generated patient: {patient.patient_id}, Age: {patient.profile.age}, Chief complaint: {patient.chief_complaint}")
        
        return patient
    
    def _generate_demographics(self, age_group: Optional[str], sex: Optional[str]) -> Dict[str, Any]:
        """Generate patient demographics."""
        demographics_data = self.demographics_config
        
        # Determine age group if not specified
        if not age_group:
            age_group = random.choice(["young_adult", "middle_aged", "geriatric"])
        
        # Generate age within the group
        age_config = demographics_data.get("age_groups", {}).get(age_group, {"min_age": 25, "max_age": 65})
        age = random.randint(age_config["min_age"], age_config["max_age"])
        
        # Determine sex if not specified
        if not sex:
            sex = random.choice(["M", "F"])
        
        # Generate occupation based on age and demographics
        occupation = self._generate_occupation(age, sex)
        
        # Generate allergies (some patients have none)
        allergies = self._generate_allergies()
        
        # Generate family history
        family_history = self._generate_family_history(age_group)
        
        return {
            "age": age,
            "sex": sex,
            "age_group": age_group,
            "occupation": occupation,
            "allergies": allergies,
            "family_history": family_history
        }
    
    def _generate_occupation(self, age: int, sex: str) -> str:
        """Generate realistic occupation based on demographics."""
        occupations_by_age = {
            "young": [
                "Student", "Retail worker", "Server", "Administrative assistant",
                "Customer service representative", "Teacher", "Nurse", "Software developer"
            ],
            "middle": [
                "Manager", "Engineer", "Teacher", "Nurse", "Sales representative",
                "Accountant", "Construction worker", "Police officer", "Firefighter",
                "Real estate agent", "Business owner"
            ],
            "older": [
                "Retired teacher", "Retired nurse", "Retired engineer", "Retired",
                "Part-time consultant", "Volunteer", "Retired business owner"
            ]
        }
        
        if age < 30:
            category = "young"
        elif age < 65:
            category = "middle"
        else:
            category = "older"
        
        return random.choice(occupations_by_age[category])
    
    def _generate_allergies(self) -> List[str]:
        """Generate patient allergies."""
        # 30% of patients have no known allergies
        if random.random() < 0.3:
            return ["NKDA"]
        
        common_allergies = [
            "Penicillin", "Sulfa drugs", "Shellfish", "Peanuts", "Tree nuts",
            "Latex", "Iodine", "Aspirin", "NSAIDs", "Codeine"
        ]
        
        # Most patients have 1-2 allergies
        num_allergies = random.choices([0, 1, 2, 3], weights=[30, 50, 15, 5])[0]
        if num_allergies == 0:
            return ["NKDA"]
        
        return random.sample(common_allergies, min(num_allergies, len(common_allergies)))
    
    def _generate_family_history(self, age_group: str) -> List[str]:
        """Generate family medical history."""
        common_family_conditions = [
            "Diabetes in mother", "Hypertension in father", "Heart disease in father",
            "Breast cancer in maternal aunt", "Colon cancer in paternal grandfather",
            "Stroke in grandmother", "Alzheimer's disease in grandfather",
            "Depression in family", "Asthma in siblings"
        ]
        
        # Older patients more likely to have family history
        if age_group == "geriatric":
            num_conditions = random.choices([0, 1, 2, 3], weights=[10, 30, 40, 20])[0]
        elif age_group == "middle_aged":
            num_conditions = random.choices([0, 1, 2], weights=[20, 50, 30])[0]
        else:
            num_conditions = random.choices([0, 1, 2], weights=[40, 40, 20])[0]
        
        if num_conditions == 0:
            return ["Non-contributory"]
        
        return random.sample(common_family_conditions, min(num_conditions, len(common_family_conditions)))
    
    def _generate_medical_history(self, demographics: Dict[str, Any], complexity: str) -> List[str]:
        """Generate patient medical history based on demographics and complexity."""
        age_group = demographics["age_group"]
        age = demographics["age"]
        
        # Get age-appropriate conditions
        possible_conditions = self.age_based_conditions.get(age_group, [])
        
        # Determine number of conditions based on complexity and age
        if complexity == "simple":
            max_conditions = 1 if age < 40 else 2
        elif complexity == "moderate":
            max_conditions = 2 if age < 40 else 4
        else:  # complex
            max_conditions = 3 if age < 40 else 6
        
        # Younger patients less likely to have multiple conditions
        if age < 30:
            max_conditions = max(1, max_conditions - 1)
        
        num_conditions = random.randint(0, max_conditions)
        
        if num_conditions == 0:
            return []
        
        return random.sample(possible_conditions, min(num_conditions, len(possible_conditions)))
    
    def _generate_medications(self, medical_history: List[str]) -> List[str]:
        """Generate medications based on medical history."""
        medications = []
        
        for condition in medical_history:
            if condition in self.condition_medications:
                # Usually one medication per condition, sometimes two
                num_meds = random.choices([1, 2], weights=[80, 20])[0]
                condition_meds = random.sample(
                    self.condition_medications[condition],
                    min(num_meds, len(self.condition_medications[condition]))
                )
                medications.extend(condition_meds)
        
        return medications
    
    def _generate_social_history(self, demographics: Dict[str, Any]) -> SocialHistory:
        """Generate social history based on demographics."""
        age = demographics["age"]
        sex = demographics["sex"]
        
        # Smoking history (age and demographically appropriate)
        smoking_options = [
            "Never smoker",
            f"Former smoker, quit {random.randint(1, 10)} years ago, {random.randint(5, 30)} pack-year history",
            f"Current smoker, {random.randint(1, 2)} pack{'s' if random.random() > 0.5 else ''} per day for {random.randint(5, 20)} years"
        ]
        
        # Smoking rates vary by age
        if age < 30:
            smoking_weights = [60, 20, 20]
        elif age < 60:
            smoking_weights = [40, 40, 20]
        else:
            smoking_weights = [50, 35, 15]
        
        smoking = random.choices(smoking_options, weights=smoking_weights)[0]
        
        # Alcohol consumption
        alcohol_options = [
            "Denies alcohol use",
            "Social drinker, 2-3 drinks per week",
            "Moderate alcohol use, 1-2 drinks daily",
            "Occasional wine with dinner",
            "Drinks 4-6 beers per week"
        ]
        alcohol = random.choice(alcohol_options)
        
        # Exercise habits
        exercise_options = [
            "Sedentary lifestyle, minimal exercise",
            "Walks 2-3 times per week",
            "Exercises regularly, 3-4 times per week",
            "Active lifestyle with daily exercise",
            "Limited by medical conditions"
        ]
        exercise = random.choice(exercise_options)
        
        return SocialHistory(
            smoking=smoking,
            alcohol=alcohol,
            exercise=exercise,
            occupation=demographics["occupation"],
            marital_status=random.choice(["Single", "Married", "Divorced", "Widowed"]),
            living_situation=random.choice(["Lives alone", "Lives with spouse", "Lives with family"])
        )
    
    def _generate_symptoms(
        self, 
        patient_profile: PatientProfile, 
        symptom_theme: Optional[str], 
        complexity: str
    ) -> Tuple[str, str, List[str]]:
        """Generate chief complaint, symptom history, and current symptoms."""
        
        # Select theme if not specified
        if not symptom_theme:
            themes = list(self.symptoms_config.get("themes", {}).keys())
            symptom_theme = random.choice(themes)
        
        theme_config = self.symptoms_config.get("themes", {}).get(symptom_theme, {})
        
        # Select primary symptom
        primary_symptoms = theme_config.get("primary_symptoms", ["Fatigue"])
        chief_complaint = random.choice(primary_symptoms).replace("_", " ").title()
        
        # Add duration and context
        durations = ["2 hours", "6 hours", "1 day", "3 days", "1 week", "2 weeks", "1 month"]
        duration = random.choice(durations)
        chief_complaint = f"{chief_complaint} for the past {duration}"
        
        # Generate current symptoms (primary + associated)
        current_symptoms = [chief_complaint.split(" for ")[0]]
        
        # Add associated symptoms based on complexity
        associated_symptoms = theme_config.get("associated_symptoms", [])
        if complexity == "simple":
            num_associated = random.randint(0, 1)
        elif complexity == "moderate":
            num_associated = random.randint(1, 3)
        else:  # complex
            num_associated = random.randint(2, 4)
        
        if associated_symptoms and num_associated > 0:
            selected_associated = random.sample(
                associated_symptoms, 
                min(num_associated, len(associated_symptoms))
            )
            current_symptoms.extend([s.replace("_", " ").title() for s in selected_associated])
        
        # Generate detailed symptom history
        symptom_history = self._generate_symptom_history(
            chief_complaint, current_symptoms, patient_profile, complexity
        )
        
        return chief_complaint, symptom_history, current_symptoms
    
    def _generate_symptom_history(
        self, 
        chief_complaint: str, 
        current_symptoms: List[str], 
        patient_profile: PatientProfile, 
        complexity: str
    ) -> str:
        """Generate detailed symptom history narrative."""
        
        primary_symptom = chief_complaint.split(" for ")[0].lower()
        
        # Basic symptom description templates
        descriptions = {
            "chest pain": [
                "substernal chest discomfort described as pressure-like",
                "sharp stabbing chest pain localized to the left side",
                "dull aching chest pain that worsens with movement"
            ],
            "shortness of breath": [
                "progressive dyspnea on exertion",
                "sudden onset shortness of breath at rest",
                "gradually worsening breathing difficulty"
            ],
            "headache": [
                "throbbing headache predominantly in the temples",
                "constant dull headache across the forehead",
                "severe pounding headache behind the eyes"
            ],
            "abdominal pain": [
                "crampy abdominal pain in the lower right quadrant",
                "burning epigastric discomfort",
                "diffuse abdominal pain with intermittent sharp episodes"
            ]
        }
        
        # Get appropriate description
        base_description = "reports symptoms consistent with the chief complaint"
        for key, desc_list in descriptions.items():
            if key in primary_symptom:
                base_description = random.choice(desc_list)
                break
        
        # Build symptom history narrative
        history_parts = [
            f"Patient reports {base_description}.",
        ]
        
        # Add timing and progression
        progressions = [
            "Symptoms began gradually and have been worsening",
            "Onset was sudden while the patient was at rest",
            "Symptoms started after physical activity",
            "Patient noticed symptoms upon waking this morning"
        ]
        history_parts.append(random.choice(progressions))
        
        # Add associated symptoms if present
        if len(current_symptoms) > 1:
            associated = current_symptoms[1:]  # Skip primary symptom
            if len(associated) == 1:
                history_parts.append(f"Associated with {associated[0].lower()}.")
            else:
                history_parts.append(f"Associated with {', '.join(associated[:-1]).lower()}, and {associated[-1].lower()}.")
        
        # Add aggravating/alleviating factors for moderate/complex cases
        if complexity in ["moderate", "complex"]:
            aggravating = [
                "worse with activity", "worse when lying down", "worse with deep breathing",
                "worse with movement", "worse in the morning", "worse at night"
            ]
            alleviating = [
                "improves with rest", "better with sitting up", "improved with over-the-counter pain medication",
                "somewhat better after eating", "improves with heat application"
            ]
            
            history_parts.append(f"Symptoms are {random.choice(aggravating)} and {random.choice(alleviating)}.")
        
        # Add relevant negatives for complex cases
        if complexity == "complex":
            negatives = [
                "Denies fever, chills, or night sweats",
                "No recent travel or sick contacts",
                "No recent changes in medications",
                "Denies weight loss or loss of appetite"
            ]
            history_parts.append(random.choice(negatives))
        
        return " ".join(history_parts)
    
    def _generate_symptom_severity(self, symptoms: List[str]) -> Dict[str, int]:
        """Generate severity ratings for symptoms on 1-10 scale."""
        severity_ratings = {}
        
        for symptom in symptoms:
            # Pain-related symptoms typically higher severity
            if any(pain_word in symptom.lower() for pain_word in ["pain", "ache", "burning", "cramping"]):
                severity = random.randint(4, 9)
            else:
                severity = random.randint(2, 7)
            
            severity_ratings[symptom] = severity
        
        return severity_ratings
    
    def generate_vital_signs(self, patient_profile: PatientProfile, acuity: str = "stable") -> VitalSigns:
        """
        Generate realistic vital signs based on patient profile and acuity.
        
        Args:
            patient_profile: Patient demographic information
            acuity: Patient acuity (stable, urgent, critical)
        
        Returns:
            Generated VitalSigns
        """
        age = patient_profile.age
        
        # Base normal ranges adjusted for age
        if acuity == "stable":
            temp_range = (98.0, 99.5)
            hr_range = (60, 100) if age < 65 else (65, 95)
            bp_sys_range = (110, 140) if age < 50 else (120, 150)
            bp_dias_range = (70, 90)
            rr_range = (12, 20)
            spo2_range = (95, 100)
        elif acuity == "urgent":
            temp_range = (99.0, 101.5)
            hr_range = (90, 120)
            bp_sys_range = (90, 160)
            bp_dias_range = (60, 100)
            rr_range = (16, 24)
            spo2_range = (92, 98)
        else:  # critical
            temp_range = (100.0, 103.0)
            hr_range = (100, 140)
            bp_sys_range = (80, 180)
            bp_dias_range = (50, 110)
            rr_range = (20, 30)
            spo2_range = (88, 95)
        
        # Generate realistic BMI and calculate weight/height
        bmi = random.uniform(18.5, 35.0)
        height_inches = random.uniform(60, 75) if patient_profile.sex == "M" else random.uniform(58, 70)
        weight_lbs = bmi * (height_inches ** 2) / 703
        
        return VitalSigns(
            temperature_f=round(random.uniform(*temp_range), 1),
            blood_pressure_systolic=random.randint(*bp_sys_range),
            blood_pressure_diastolic=random.randint(*bp_dias_range),
            heart_rate=random.randint(*hr_range),
            respiratory_rate=random.randint(*rr_range),
            oxygen_saturation=round(random.uniform(*spo2_range), 1),
            weight_lbs=round(weight_lbs, 1),
            height_inches=round(height_inches, 1),
            bmi=round(bmi, 1)
        )