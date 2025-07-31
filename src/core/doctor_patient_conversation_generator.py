"""
Doctor-Patient Conversation Generator for synthetic medical dialogues.

This module creates realistic multi-turn conversations between doctors and patients,
following natural consultation patterns and incorporating medical interview techniques,
patient communication styles, and clinical workflows.
"""

import asyncio
import os
import random
import time
import yaml
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import openai
from openai import OpenAI

from models import (
    Doctor, Patient, ConversationTurn, Conversation, MedicalExpert,
    PatientProfile, MedicalCase
)
from utils.logging_setup import get_logger


class DoctorPatientConversationGenerator:
    """
    Generates realistic doctor-patient conversations for medical case creation.
    
    This class creates natural, multi-turn conversations that follow clinical
    consultation patterns while incorporating patient communication styles,
    medical interview techniques, and realistic dialogue flow.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        medical_model: str = "gpt-4-turbo",
        config_path: Optional[str] = None,
        rate_limit_delay: float = 1.0
    ):
        """
        Initialize the Conversation Generator.
        
        Args:
            openai_api_key: OpenAI API key (if None, will use environment variable)
            medical_model: OpenAI model for conversation generation
            config_path: Path to configuration files
            rate_limit_delay: Delay between API calls for rate limiting
        """
        self.logger = get_logger(f"{__name__}.DoctorPatientConversationGenerator")
        
        # Setup OpenAI client
        self.client = OpenAI(
            api_key=openai_api_key or os.getenv('OPENAI_API_KEY')
        )
        self.medical_model = medical_model
        self.rate_limit_delay = rate_limit_delay
        
        # Load configuration
        self.config_path = Path(config_path) if config_path else Path(__file__).parent.parent / "config" / "prompts"
        self.conversation_config = self._load_yaml_config("conversation_flows.yaml")
        self.demographics_config = self._load_yaml_config("demographic_variations.yaml")
        
        self.logger.info("DoctorPatientConversationGenerator initialized successfully")
    
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
    
    async def generate_conversation(
        self,
        doctor: MedicalExpert,
        patient: Patient,
        consultation_type: str = "initial",
        target_length: int = 15  # Target number of conversation turns
    ) -> Conversation:
        """
        Generate a complete doctor-patient conversation.
        
        Args:
            doctor: Medical expert conducting the consultation
            patient: Patient profile and presentation
            consultation_type: Type of consultation (initial, follow_up, urgent)
            target_length: Target number of conversation turns
        
        Returns:
            Generated Conversation with complete transcript
        """
        start_time = time.time()
        
        self.logger.info(
            f"Generating doctor-patient conversation",
            extra={
                'doctor_id': doctor.doctor.doctor_id,
                'patient_id': patient.patient_id,
                'consultation_type': consultation_type,
                'target_length': target_length,
                'operation': 'conversation_generation'
            }
        )
        
        try:
            # Generate conversation phases
            conversation_turns = await self._generate_conversation_phases(
                doctor, patient, consultation_type, target_length
            )
            
            # Calculate approximate duration
            estimated_duration = self._estimate_consultation_duration(conversation_turns)
            
            # Create conversation object
            conversation = Conversation(
                transcript=conversation_turns,
                consultation_type=consultation_type,
                duration_minutes=estimated_duration,
                consultation_setting=self._determine_consultation_setting(consultation_type)
            )
            
            # Log successful generation
            duration = time.time() - start_time
            self.logger.info(
                "Doctor-patient conversation generated successfully",
                extra={
                    'doctor_id': doctor.doctor.doctor_id,
                    'patient_id': patient.patient_id,
                    'conversation_turns': len(conversation_turns),
                    'estimated_duration_minutes': estimated_duration,
                    'generation_duration': duration,
                    'operation': 'conversation_complete'
                }
            )
            
            return conversation
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                f"Conversation generation failed: {str(e)}",
                extra={
                    'doctor_id': doctor.doctor.doctor_id,
                    'patient_id': patient.patient_id,
                    'generation_duration': duration,
                    'error_type': type(e).__name__,
                    'operation': 'conversation_error'
                },
                exc_info=True
            )
            raise
    
    async def _generate_conversation_phases(
        self,
        doctor: MedicalExpert,
        patient: Patient,
        consultation_type: str,
        target_length: int
    ) -> List[ConversationTurn]:
        """Generate conversation following consultation phases."""
        
        conversation_turns = []
        current_time = "0:00"
        turn_count = 0
        
        # Get consultation phases from config
        phases = self.conversation_config.get("consultation_phases", {})
        
        # Phase order for different consultation types
        if consultation_type == "initial":
            phase_order = ["opening", "chief_complaint", "history_of_present_illness", 
                          "past_medical_history", "social_history", "assessment_and_plan", "closing"]
        elif consultation_type == "follow_up":
            phase_order = ["opening", "interval_history", "assessment_and_plan", "closing"]
        else:  # urgent
            phase_order = ["opening", "chief_complaint", "focused_assessment", "immediate_plan", "closing"]
        
        for phase_name in phase_order:
            if turn_count >= target_length:
                break
                
            phase_config = phases.get(phase_name, {})
            phase_turns = await self._generate_phase_conversation(
                doctor, patient, phase_name, phase_config, current_time
            )
            
            conversation_turns.extend(phase_turns)
            turn_count += len(phase_turns)
            
            # Update current time
            current_time = self._advance_time(current_time, phase_config.get("duration_range", [2, 5]))
        
        return conversation_turns[:target_length]  # Trim to target length
    
    async def _generate_phase_conversation(
        self,
        doctor: MedicalExpert,
        patient: Patient,
        phase_name: str,
        phase_config: Dict[str, Any],
        start_time: str
    ) -> List[ConversationTurn]:
        """Generate conversation for a specific consultation phase."""
        
        await asyncio.sleep(self.rate_limit_delay)
        
        # Build phase-specific prompt
        prompt = self._build_phase_prompt(doctor, patient, phase_name, phase_config)
        
        try:
            response = self.client.chat.completions.create(
                model=self.medical_model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_conversation_system_prompt(doctor, patient, phase_name)
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.8,  # Higher temperature for more natural conversation
                max_tokens=1000
            )
            
            conversation_text = response.choices[0].message.content
            
            # Parse the generated conversation into turns
            turns = self._parse_conversation_text(conversation_text, start_time, phase_name)
            
            return turns
            
        except Exception as e:
            self.logger.error(f"Phase conversation generation failed for {phase_name}: {str(e)}")
            # Return fallback conversation
            return self._generate_fallback_conversation(doctor, patient, phase_name, start_time)
    
    def _build_phase_prompt(
        self,
        doctor: MedicalExpert,
        patient: Patient,
        phase_name: str,
        phase_config: Dict[str, Any]
    ) -> str:
        """Build conversation prompt for specific phase."""
        
        prompt_parts = [
            f"Generate a realistic conversation for the {phase_name.replace('_', ' ')} phase of a medical consultation.",
            "",
            f"Doctor Profile:",
            f"- Specialty: {doctor.doctor.specialty.primary_specialty}",
            f"- Experience: {doctor.doctor.experience.years_of_practice} years",
            f"- Communication Style: {doctor.doctor.communication_style.bedside_manner}",
            f"- Questioning Approach: {doctor.doctor.communication_style.questioning_approach}",
            "",
            f"Patient Profile:",
            f"- Age: {patient.profile.age}, Sex: {patient.profile.sex}",
            f"- Occupation: {patient.profile.occupation}",
            f"- Chief Complaint: {patient.chief_complaint}",
            f"- Medical History: {', '.join(patient.profile.medical_history) if patient.profile.medical_history else 'None'}",
            "",
            f"Phase Focus:"
        ]
        
        # Add phase-specific guidance
        doctor_actions = phase_config.get("doctor_actions", [])
        patient_responses = phase_config.get("patient_responses", [])
        
        if doctor_actions:
            prompt_parts.append("Doctor should:")
            prompt_parts.extend([f"- {action.replace('_', ' ').title()}" for action in doctor_actions])
        
        if patient_responses:
            prompt_parts.append("Patient should:")
            prompt_parts.extend([f"- {response.replace('_', ' ').title()}" for response in patient_responses])
        
        prompt_parts.extend([
            "",
            "Generate 2-4 conversation turns for this phase.",
            "Format: 'Doctor: [statement]' and 'Patient: [response]' on separate lines.",
            "Make the conversation natural and realistic."
        ])
        
        return "\n".join(prompt_parts)
    
    def _get_conversation_system_prompt(
        self,
        doctor: MedicalExpert,
        patient: Patient,
        phase_name: str
    ) -> str:
        """Get system prompt for conversation generation."""
        
        return f"""You are generating a realistic medical consultation conversation between a doctor and patient.

Doctor characteristics:
- {doctor.doctor.specialty.primary_specialty} specialist
- {doctor.doctor.communication_style.bedside_manner} communication style
- {doctor.doctor.communication_style.explanation_style} explanation approach
- {doctor.doctor.experience.years_of_practice} years of experience

Patient characteristics:
- {patient.profile.age}-year-old {patient.profile.sex.lower()}
- Occupation: {patient.profile.occupation}
- Presenting with: {patient.chief_complaint}

Generate natural, realistic dialogue that:
1. Reflects the doctor's professional communication style
2. Shows appropriate patient responses based on demographics and health literacy
3. Follows medical interview best practices
4. Includes natural conversation flow with appropriate interruptions and clarifications
5. Maintains clinical accuracy and professionalism

Focus on the {phase_name.replace('_', ' ')} phase of the consultation."""
    
    def _parse_conversation_text(
        self,
        conversation_text: str,
        start_time: str,
        phase_name: str
    ) -> List[ConversationTurn]:
        """Parse generated conversation text into ConversationTurn objects."""
        
        turns = []
        lines = conversation_text.strip().split('\n')
        current_time = start_time
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Parse doctor and patient statements
            if line.startswith("Doctor:"):
                content = line.replace("Doctor:", "").strip()
                if content:
                    turns.append(ConversationTurn(
                        speaker="doctor",
                        content=content,
                        timestamp=current_time,
                        intent=self._determine_intent(content, "doctor", phase_name)
                    ))
                    current_time = self._advance_time(current_time, [0.5, 1.5])
                    
            elif line.startswith("Patient:"):
                content = line.replace("Patient:", "").strip()
                if content:
                    turns.append(ConversationTurn(
                        speaker="patient",
                        content=content,
                        timestamp=current_time,
                        intent=self._determine_intent(content, "patient", phase_name)
                    ))
                    current_time = self._advance_time(current_time, [0.5, 1.5])
        
        return turns
    
    def _determine_intent(self, content: str, speaker: str, phase_name: str) -> str:
        """Determine the intent of a conversation turn."""
        
        content_lower = content.lower()
        
        if speaker == "doctor":
            if any(q in content for q in ["?", "tell me", "describe", "how", "when", "where", "what"]):
                return "question"
            elif any(e in content_lower for e in ["let me explain", "this means", "what this suggests"]):
                return "explanation"
            elif any(i in content_lower for i in ["i recommend", "you should", "let's", "we need to"]):
                return "instruction"
            else:
                return "clarification"
        else:  # patient
            if "?" in content:
                return "question"
            elif any(c in content_lower for c in ["i'm worried", "i'm concerned", "what if"]):
                return "concern"
            else:
                return "answer"
    
    def _advance_time(self, current_time: str, duration_range: List[float]) -> str:
        """Advance the conversation timestamp."""
        
        # Parse current time
        try:
            if ":" in current_time:
                minutes, seconds = map(int, current_time.split(":"))
                total_seconds = minutes * 60 + seconds
            else:
                total_seconds = 0
        except:
            total_seconds = 0
        
        # Add random duration
        additional_seconds = random.uniform(duration_range[0], duration_range[1]) * 60
        total_seconds += int(additional_seconds)
        
        # Convert back to mm:ss format
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        
        return f"{minutes}:{seconds:02d}"
    
    def _estimate_consultation_duration(self, turns: List[ConversationTurn]) -> int:
        """Estimate total consultation duration from conversation turns."""
        
        if not turns:
            return 15  # Default duration
        
        # Base duration calculation
        base_duration = len(turns) * 1.5  # ~1.5 minutes per turn average
        
        # Adjust based on conversation content complexity
        complex_indicators = sum(1 for turn in turns if len(turn.content.split()) > 20)
        duration_adjustment = complex_indicators * 2
        
        total_duration = int(base_duration + duration_adjustment)
        
        # Reasonable bounds
        return max(10, min(60, total_duration))
    
    def _determine_consultation_setting(self, consultation_type: str) -> str:
        """Determine consultation setting based on type."""
        
        settings = {
            "initial": "Primary care office",
            "follow_up": "Primary care office", 
            "urgent": "Urgent care clinic",
            "emergency": "Emergency department",
            "specialty": "Specialty clinic"
        }
        
        return settings.get(consultation_type, "Medical office")
    
    def _generate_fallback_conversation(
        self,
        doctor: MedicalExpert,
        patient: Patient,
        phase_name: str,
        start_time: str
    ) -> List[ConversationTurn]:
        """Generate fallback conversation if API call fails."""
        
        fallback_conversations = {
            "opening": [
                ("doctor", "Good morning, I'm Dr. Smith. What brings you in today?"),
                ("patient", f"Hello Doctor. I've been having {patient.chief_complaint.lower()}.")
            ],
            "chief_complaint": [
                ("doctor", "Can you tell me more about your symptoms?"),
                ("patient", f"Well, {patient.symptom_history}"),
                ("doctor", "I see. How long has this been going on?"),
                ("patient", "It started a few days ago and seems to be getting worse.")
            ]
        }
        
        default_turns = [
            ("doctor", "Thank you for that information."),
            ("patient", "Is there anything else you need to know?")
        ]
        
        conversation_template = fallback_conversations.get(phase_name, default_turns)
        
        turns = []
        current_time = start_time
        
        for speaker, content in conversation_template:
            turns.append(ConversationTurn(
                speaker=speaker,
                content=content,
                timestamp=current_time,
                intent="question" if speaker == "doctor" else "answer"
            ))
            current_time = self._advance_time(current_time, [0.5, 1.5])
        
        return turns
    
    def generate_patient_education_dialogue(
        self,
        doctor: MedicalExpert,
        patient: Patient,
        diagnosis: str,
        treatment_plan: str
    ) -> List[ConversationTurn]:
        """
        Generate patient education dialogue for diagnosis explanation.
        
        Args:
            doctor: Medical expert explaining the condition
            patient: Patient receiving education
            diagnosis: Final diagnosis to explain
            treatment_plan: Treatment plan to discuss
        
        Returns:
            List of conversation turns for patient education
        """
        
        # This could be expanded to use OpenAI for dynamic generation
        # For now, using template-based approach for reliability
        
        education_turns = []
        
        # Doctor explains diagnosis
        education_turns.append(ConversationTurn(
            speaker="doctor",
            content=f"Based on our examination and your symptoms, I believe you have {diagnosis}. Let me explain what this means.",
            timestamp="end_consultation",
            intent="explanation"
        ))
        
        # Patient asks for clarification
        education_turns.append(ConversationTurn(
            speaker="patient", 
            content="What does that mean exactly? Is it serious?",
            timestamp="end_consultation",
            intent="question"
        ))
        
        # Doctor provides education
        education_turns.append(ConversationTurn(
            speaker="doctor",
            content=f"This condition is manageable with proper treatment. Here's what we're going to do: {treatment_plan}",
            timestamp="end_consultation", 
            intent="explanation"
        ))
        
        return education_turns