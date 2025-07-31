"""
Basic functionality tests for Medical Dataset Generator.

These tests verify that the core components can be initialized and basic
operations work correctly without requiring API calls.
"""

import unittest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.patient_generator import PatientGenerator
from core.medical_expert_generator import MedicalExpertGenerator
from models import Patient, Doctor, MedicalExpert


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality without API calls."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.patient_generator = PatientGenerator()
        # Don't initialize MedicalExpertGenerator to avoid API key requirement
    
    def test_patient_generator_initialization(self):
        """Test that PatientGenerator initializes correctly."""
        self.assertIsInstance(self.patient_generator, PatientGenerator)
        self.assertIsNotNone(self.patient_generator.demographics_config)
    
    def test_patient_generation(self):
        """Test basic patient generation."""
        patient = self.patient_generator.generate_patient(
            age_group="middle_aged",
            sex="M",
            symptom_theme="cardiovascular",
            complexity="moderate"
        )
        
        # Verify patient structure
        self.assertIsInstance(patient, Patient)
        self.assertIsNotNone(patient.patient_id)
        self.assertIsNotNone(patient.profile)
        self.assertIsNotNone(patient.chief_complaint)
        self.assertIsNotNone(patient.symptom_history)
        
        # Verify demographics
        self.assertEqual(patient.profile.sex, "M")
        self.assertGreaterEqual(patient.profile.age, 36)  # middle_aged range
        self.assertLessEqual(patient.profile.age, 64)
    
    def test_patient_demographics_variations(self):
        """Test that patient demographics vary appropriately."""
        patients = []
        for i in range(5):
            patient = self.patient_generator.generate_patient(complexity="simple")
            patients.append(patient)
        
        # Verify we get different patients
        patient_ids = [p.patient_id for p in patients]
        self.assertEqual(len(set(patient_ids)), 5)  # All unique IDs
        
        # Verify demographic variation
        ages = [p.profile.age for p in patients]
        self.assertGreater(max(ages) - min(ages), 5)  # Some age variation
    
    def test_medical_expert_generation_without_api(self):
        """Test medical expert generation without API calls."""
        # This test doesn't require API calls, just model creation
        expert_generator = MedicalExpertGenerator(openai_api_key="dummy_key")
        
        expert = expert_generator.generate_medical_expert(
            specialty="cardiology",
            experience_level="senior",
            communication_style="empathetic"
        )
        
        # Verify expert structure
        self.assertIsInstance(expert, MedicalExpert)
        self.assertIsInstance(expert.doctor, Doctor)
        self.assertEqual(expert.doctor.specialty.primary_specialty, "cardiology")
        self.assertEqual(expert.doctor.communication_style.bedside_manner, "empathetic")
        self.assertGreaterEqual(expert.doctor.experience.years_of_practice, 21)  # senior level
    
    def test_vital_signs_generation(self):
        """Test vital signs generation."""
        patient = self.patient_generator.generate_patient(complexity="simple")
        
        vital_signs = self.patient_generator.generate_vital_signs(
            patient.profile, 
            acuity="stable"
        )
        
        # Verify vital signs are within reasonable ranges
        self.assertIsNotNone(vital_signs.temperature_f)
        self.assertGreaterEqual(vital_signs.temperature_f, 98.0)
        self.assertLessEqual(vital_signs.temperature_f, 99.5)
        
        self.assertIsNotNone(vital_signs.heart_rate)
        self.assertGreaterEqual(vital_signs.heart_rate, 60)
        self.assertLessEqual(vital_signs.heart_rate, 100)
        
        self.assertIsNotNone(vital_signs.blood_pressure_systolic)
        self.assertGreaterEqual(vital_signs.blood_pressure_systolic, 110)
        
        self.assertIsNotNone(vital_signs.bmi)
        self.assertGreater(vital_signs.bmi, 18.0)
        self.assertLess(vital_signs.bmi, 40.0)
    
    def test_symptom_severity_generation(self):
        """Test symptom severity rating generation."""
        symptoms = ["Chest pain", "Shortness of breath", "Fatigue"]
        
        severity_ratings = self.patient_generator._generate_symptom_severity(symptoms)
        
        # Verify all symptoms have ratings
        self.assertEqual(len(severity_ratings), 3)
        
        # Verify ratings are in valid range
        for symptom, rating in severity_ratings.items():
            self.assertIn(symptom, symptoms)
            self.assertGreaterEqual(rating, 1)
            self.assertLessEqual(rating, 10)
        
        # Pain symptoms should generally have higher ratings
        if "Chest pain" in severity_ratings:
            self.assertGreaterEqual(severity_ratings["Chest pain"], 4)
    
    def test_config_loading(self):
        """Test that configuration files are loaded correctly."""
        # Test demographics config
        demographics = self.patient_generator.demographics_config
        self.assertIn("age_groups", demographics)
        self.assertIn("sex_variations", demographics)
        
        # Test age groups
        age_groups = demographics["age_groups"]
        self.assertIn("middle_aged", age_groups)
        self.assertIn("geriatric", age_groups)
        
        # Test symptoms config if available
        symptoms = self.patient_generator.symptoms_config
        if symptoms:
            self.assertIn("themes", symptoms)


class TestDataModels(unittest.TestCase):
    """Test data model validation and serialization."""
    
    def test_patient_model_validation(self):
        """Test patient model Pydantic validation."""
        from models import PatientProfile, SocialHistory
        
        # Valid patient profile
        social_history = SocialHistory(
            smoking="Never smoker",
            alcohol="Social drinker",
            exercise="Regular exercise"
        )
        
        profile = PatientProfile(
            age=45,
            sex="M",
            occupation="Engineer",
            medical_history=["Hypertension"],
            medications=["Lisinopril 10mg daily"],
            allergies=["NKDA"],
            family_history=["Diabetes in father"],
            social_history=social_history
        )
        
        # Verify profile creation
        self.assertEqual(profile.age, 45)
        self.assertEqual(profile.sex, "M")
        self.assertIn("Hypertension", profile.medical_history)
    
    def test_invalid_patient_data(self):
        """Test that invalid patient data raises validation errors."""
        from models import PatientProfile, SocialHistory
        from pydantic import ValidationError
        
        social_history = SocialHistory(
            smoking="Never smoker",
            alcohol="Social drinker", 
            exercise="Regular exercise"
        )
        
        # Invalid age (too high)
        with self.assertRaises(ValidationError):
            PatientProfile(
                age=150,  # Invalid age
                sex="M",
                occupation="Engineer",
                social_history=social_history
            )
        
        # Invalid sex
        with self.assertRaises(ValidationError):
            PatientProfile(
                age=45,
                sex="Invalid",  # Invalid sex value
                occupation="Engineer", 
                social_history=social_history
            )


if __name__ == "__main__":
    unittest.main()