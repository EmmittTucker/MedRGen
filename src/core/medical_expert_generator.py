"""
Medical Expert Generator for synthetic medical reasoning.

This module creates medical experts with different specialties and generates
specialty-specific medical reasoning using OpenAI models. It produces
evidence-based diagnostic thinking, treatment planning, and clinical decision-making.
"""

import os
import asyncio
import yaml
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import openai
from openai import OpenAI
import time
import random

from models import (
    Doctor, MedicalExpert, MedicalSpecialty, CommunicationStyle, 
    ClinicalExperience, ReasoningPattern, MedicalCase, PatientProfile
)
from utils.logging_setup import get_logger


class MedicalExpertGenerator:
    """
    Generates medical experts and provides specialty-specific medical reasoning.
    
    This class creates diverse medical experts with different specialties,
    experience levels, and reasoning patterns, then uses them to generate
    high-quality medical reasoning for synthetic case creation.
    """
    
    def __init__(
        self, 
        openai_api_key: Optional[str] = None,
        medical_model: str = "gpt-4-turbo",
        config_path: Optional[str] = None,
        rate_limit_delay: float = 1.0
    ):
        """
        Initialize the Medical Expert Generator.
        
        Args:
            openai_api_key: OpenAI API key (if None, will use environment variable)
            medical_model: OpenAI model for medical reasoning
            config_path: Path to configuration files
            rate_limit_delay: Delay between API calls for rate limiting
        """
        self.logger = get_logger(f"{__name__}.MedicalExpertGenerator")
        
        # Setup OpenAI client
        self.client = OpenAI(
            api_key=openai_api_key or os.getenv('OPENAI_API_KEY')
        )
        self.medical_model = medical_model
        self.rate_limit_delay = rate_limit_delay
        
        # Load configuration
        self.config_path = Path(config_path) if config_path else Path(__file__).parent.parent / "config" / "prompts"
        self.specialties_config = self._load_yaml_config("medical_specialties.yaml")
        self.reasoning_config = self._load_yaml_config("reasoning_constraints.yaml")
        
        # Cache for generated experts
        self.expert_cache: Dict[str, MedicalExpert] = {}
        
        self.logger.info("MedicalExpertGenerator initialized successfully")
    
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
    
    def generate_medical_expert(
        self, 
        specialty: str = "internal_medicine",
        experience_level: str = "moderate",
        communication_style: str = "empathetic"
    ) -> MedicalExpert:
        """
        Generate a medical expert with specified characteristics.
        
        Args:
            specialty: Medical specialty
            experience_level: Experience level (junior, moderate, senior)
            communication_style: Communication approach
        
        Returns:
            Generated MedicalExpert instance
        """
        cache_key = f"{specialty}_{experience_level}_{communication_style}"
        
        if cache_key in self.expert_cache:
            self.logger.debug(f"Returning cached expert: {cache_key}")
            return self.expert_cache[cache_key]
        
        self.logger.info(f"Generating new medical expert: {specialty}, {experience_level}, {communication_style}")
        
        # Get specialty configuration
        specialty_config = self.specialties_config.get("specialties", {}).get(specialty, {})
        
        # Generate years of practice based on experience level
        years_map = {"junior": (2, 7), "moderate": (8, 20), "senior": (21, 40)}
        min_years, max_years = years_map.get(experience_level, (8, 20))
        years_of_practice = random.randint(min_years, max_years)
        
        # Create doctor profile
        doctor = Doctor(
            doctor_id=f"DOC_{specialty.upper()}_{random.randint(1000, 9999)}",
            title="MD",
            specialty=MedicalSpecialty(
                primary_specialty=specialty,
                subspecialties=specialty_config.get("focus_areas", [])[:2],
                board_certifications=[specialty.replace("_", " ").title()],
                areas_of_expertise=specialty_config.get("common_presentations", [])
            ),
            experience=ClinicalExperience(
                years_of_practice=years_of_practice,
                practice_setting="Academic medical center" if experience_level == "senior" else "Community hospital",
                patient_volume="high" if experience_level == "senior" else "moderate",
                case_complexity="complex" if experience_level == "senior" else "mixed",
                teaching_experience=experience_level == "senior",
                research_background=experience_level == "senior"
            ),
            communication_style=CommunicationStyle(
                bedside_manner=communication_style,
                explanation_style="layperson_friendly" if communication_style == "empathetic" else "professional",
                questioning_approach="systematic",
                decision_making_style="evidence_based"
            ),
            reasoning_pattern=ReasoningPattern(
                diagnostic_approach="hypothetico_deductive",
                risk_tolerance="moderate",
                strengths=["systematic_thinking", "evidence_evaluation"],
                cognitive_biases=["anchoring_bias"] if experience_level == "junior" else []
            )
        )
        
        # Create medical expert with generation parameters
        expert = MedicalExpert(
            doctor=doctor,
            generation_parameters={
                "temperature": 0.7,
                "max_tokens": 2000,
                "reasoning_depth": "detailed" if experience_level == "senior" else "moderate",
                "uncertainty_acknowledgment": True,
                "evidence_citation": experience_level == "senior"
            },
            knowledge_base_focus=specialty_config.get("focus_areas", []),
            case_type_preferences=specialty_config.get("common_presentations", [])
        )
        
        # Cache the expert
        self.expert_cache[cache_key] = expert
        
        self.logger.info(f"Generated medical expert: {expert.doctor.doctor_id}")
        return expert
    
    async def generate_medical_reasoning(
        self,
        expert: MedicalExpert,
        patient_profile: PatientProfile,
        chief_complaint: str,
        symptom_history: str,
        reasoning_type: str = "diagnostic_workup"
    ) -> str:
        """
        Generate medical reasoning using the specified expert.
        
        Args:
            expert: Medical expert to use for reasoning
            patient_profile: Patient demographic and medical information
            chief_complaint: Primary reason for visit
            symptom_history: Detailed symptom progression
            reasoning_type: Type of reasoning (diagnostic_workup, differential_diagnosis, treatment_plan)
        
        Returns:
            Generated medical reasoning text
        """
        start_time = time.time()
        
        self.logger.info(
            f"Generating medical reasoning",
            extra={
                'doctor_id': expert.doctor.doctor_id,
                'specialty': expert.doctor.specialty.primary_specialty,
                'reasoning_type': reasoning_type,
                'operation': 'medical_reasoning_generation'
            }
        )
        
        # Build reasoning prompt
        prompt = self._build_reasoning_prompt(
            expert, patient_profile, chief_complaint, symptom_history, reasoning_type
        )
        
        try:
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            # Generate reasoning using OpenAI
            response = self.client.chat.completions.create(
                model=self.medical_model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt(expert, reasoning_type)
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=expert.generation_parameters.get("temperature", 0.7),
                max_tokens=expert.generation_parameters.get("max_tokens", 2000)
            )
            
            reasoning = response.choices[0].message.content
            
            # Log successful generation
            duration = time.time() - start_time
            self.logger.info(
                "Medical reasoning generated successfully",
                extra={
                    'doctor_id': expert.doctor.doctor_id,
                    'reasoning_type': reasoning_type,
                    'duration': duration,
                    'tokens_used': response.usage.total_tokens,
                    'operation': 'medical_reasoning_complete'
                }
            )
            
            return reasoning
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                f"Medical reasoning generation failed: {str(e)}",
                extra={
                    'doctor_id': expert.doctor.doctor_id,
                    'reasoning_type': reasoning_type,
                    'duration': duration,
                    'error_type': type(e).__name__,
                    'operation': 'medical_reasoning_error'
                },
                exc_info=True
            )
            raise
    
    def _build_reasoning_prompt(
        self,
        expert: MedicalExpert,
        patient_profile: PatientProfile,
        chief_complaint: str,
        symptom_history: str,
        reasoning_type: str
    ) -> str:
        """Build the reasoning prompt for the medical expert."""
        
        prompt_parts = [
            f"Patient Information:",
            f"Age: {patient_profile.age}, Sex: {patient_profile.sex}",
            f"Occupation: {patient_profile.occupation}",
            f"Medical History: {', '.join(patient_profile.medical_history) if patient_profile.medical_history else 'None'}",
            f"Medications: {', '.join(patient_profile.medications) if patient_profile.medications else 'None'}",
            f"Allergies: {', '.join(patient_profile.allergies) if patient_profile.allergies else 'NKDA'}",
            f"Family History: {', '.join(patient_profile.family_history) if patient_profile.family_history else 'Non-contributory'}",
            f"Social History: Smoking - {patient_profile.social_history.smoking}, Alcohol - {patient_profile.social_history.alcohol}",
            "",
            f"Chief Complaint: {chief_complaint}",
            "",
            f"History of Present Illness:",
            symptom_history,
            "",
            f"Please provide {reasoning_type.replace('_', ' ')} for this patient case."
        ]
        
        if reasoning_type == "diagnostic_workup":
            prompt_parts.extend([
                "",
                "Include:",
                "1. Initial diagnostic impression",
                "2. Differential diagnoses with rationales", 
                "3. Recommended diagnostic tests",
                "4. Physical examination focus areas",
                "5. Clinical reasoning process"
            ])
        elif reasoning_type == "treatment_plan":
            prompt_parts.extend([
                "",
                "Include:",
                "1. Immediate management",
                "2. Short-term treatment plan",
                "3. Long-term management",
                "4. Monitoring parameters",
                "5. Patient education points"
            ])
        
        return "\n".join(prompt_parts)
    
    def _get_system_prompt(self, expert: MedicalExpert, reasoning_type: str) -> str:
        """Get the system prompt for the medical expert."""
        
        base_prompt = f"""You are Dr. {expert.doctor.doctor_id}, a {expert.doctor.specialty.primary_specialty} specialist with {expert.doctor.experience.years_of_practice} years of experience in {expert.doctor.experience.practice_setting}.

Your expertise includes: {', '.join(expert.doctor.specialty.areas_of_expertise)}

Your communication style is {expert.doctor.communication_style.bedside_manner} with a {expert.doctor.communication_style.explanation_style} explanation approach. You use a {expert.doctor.reasoning_pattern.diagnostic_approach} reasoning approach and have {expert.doctor.reasoning_pattern.risk_tolerance} risk tolerance.

"""
        
        if reasoning_type == "diagnostic_workup":
            base_prompt += """Provide thorough diagnostic reasoning following evidence-based medicine principles. Consider:
- Clinical presentation patterns
- Disease prevalence and epidemiology  
- Diagnostic test characteristics
- Risk stratification
- Safety considerations and red flags

Your reasoning should be systematic, thorough, and demonstrate clinical expertise."""
        
        elif reasoning_type == "treatment_plan":
            base_prompt += """Develop comprehensive treatment plans based on:
- Current clinical guidelines
- Patient-specific factors
- Risk-benefit analysis
- Monitoring requirements
- Patient education needs

Ensure recommendations are evidence-based and appropriate for the patient's condition and circumstances."""
        
        # Add specialty-specific guidance
        if expert.doctor.specialty.primary_specialty == "cardiology":
            base_prompt += "\n\nFocus on cardiovascular risk factors, cardiac examination findings, and guideline-based cardiac care."
        elif expert.doctor.specialty.primary_specialty == "emergency_medicine":
            base_prompt += "\n\nPrioritize time-sensitive diagnoses, stabilization, and disposition decisions."
        
        return base_prompt
    
    def get_available_specialties(self) -> List[str]:
        """Get list of available medical specialties."""
        return list(self.specialties_config.get("specialties", {}).keys())
    
    def get_expert_by_specialty(self, specialty: str) -> Optional[MedicalExpert]:
        """Get a cached expert by specialty, or generate a new one."""
        # Look for cached expert with this specialty
        for expert in self.expert_cache.values():
            if expert.doctor.specialty.primary_specialty == specialty:
                return expert
        
        # Generate new expert if not cached
        return self.generate_medical_expert(specialty=specialty)