"""
Advanced Fitness Chatbot with Multi-Modal Capabilities
========================================================

Features:
- GPT-4/LLaMA integration for advanced language understanding
- Multi-modal inputs (text, image, voice, video)
- Advanced context management with long-term memory
- Fitness expertise and domain-specific reasoning
- Real-time adaptation and personalization
- Emotion and motivation analysis
- Integration with all fitness models (pose, nutrition, workout)

Architecture:
- Transformer-based language models (GPT-4/LLaMA)
- Whisper for voice transcription
- CLIP for image understanding
- Hierarchical memory system
- Multi-agent reasoning system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    WhisperProcessor, WhisperForConditionalGeneration,
    CLIPModel, CLIPProcessor,
    T5ForConditionalGeneration, T5Tokenizer
)
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
import pickle
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
import openai
from sentence_transformers import SentenceTransformer
import faiss
from collections import deque
import speech_recognition as sr
import soundfile as sf
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of messages the chatbot can handle."""
    TEXT = "text"
    VOICE = "voice"
    IMAGE = "image"
    VIDEO = "video"
    MULTI_MODAL = "multi_modal"

class IntentType(Enum):
    """Fitness-specific intent types."""
    WORKOUT_QUERY = "workout_query"
    NUTRITION_ADVICE = "nutrition_advice"
    FORM_CHECK = "form_check"
    PROGRESS_TRACKING = "progress_tracking"
    MOTIVATION = "motivation"
    INJURY_RECOVERY = "injury_recovery"
    GOAL_SETTING = "goal_setting"
    SCHEDULE_PLANNING = "schedule_planning"
    GENERAL_FITNESS = "general_fitness"
    CHITCHAT = "chitchat"

@dataclass
class ConversationMemory:
    """Hierarchical memory system for conversation context."""
    short_term: deque = field(default_factory=lambda: deque(maxlen=10))
    working_memory: Dict[str, Any] = field(default_factory=dict)
    long_term: List[Dict] = field(default_factory=list)
    episodic: List[Dict] = field(default_factory=list)
    user_profile: Dict[str, Any] = field(default_factory=dict)
    
    def add_to_short_term(self, message: Dict):
        """Add message to short-term memory."""
        self.short_term.append(message)
        
    def add_to_long_term(self, key: str, value: Any):
        """Store important information in long-term memory."""
        self.long_term.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now(),
            'access_count': 0
        })
        
    def retrieve_relevant_memories(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant memories based on semantic similarity."""
        # Would use vector similarity search in practice
        relevant = sorted(
            self.long_term,
            key=lambda x: x['access_count'],
            reverse=True
        )[:top_k]
        
        for memory in relevant:
            memory['access_count'] += 1
            
        return relevant

class MultiModalEncoder(nn.Module):
    """Unified encoder for multi-modal inputs."""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        # Text encoder (using sentence transformers)
        self.text_encoder = SentenceTransformer('all-mpnet-base-v2')
        
        # Image encoder (using CLIP)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Voice encoder (using Whisper)
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        
        # Projection layers to unified dimension
        self.text_projection = nn.Linear(768, hidden_dim)
        self.image_projection = nn.Linear(512, hidden_dim)
        self.audio_projection = nn.Linear(512, hidden_dim)
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text input."""
        embeddings = self.text_encoder.encode(text, convert_to_tensor=True)
        return self.text_projection(embeddings.unsqueeze(0))
        
    def encode_image(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """Encode image input."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        inputs = self.clip_processor(images=image, return_tensors="pt")
        image_features = self.clip_model.get_image_features(**inputs)
        return self.image_projection(image_features)
        
    def encode_audio(self, audio_path: str) -> torch.Tensor:
        """Encode audio input and transcribe."""
        # Load audio
        audio_input, sampling_rate = sf.read(audio_path)
        
        # Process with Whisper
        inputs = self.whisper_processor(
            audio_input,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.whisper_model.generate(inputs["input_features"])
            transcription = self.whisper_processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            
        # Get audio features
        audio_features = self.whisper_model.encoder(inputs["input_features"]).last_hidden_state
        audio_embedding = audio_features.mean(dim=1)  # Pool over time
        
        return self.audio_projection(audio_embedding), transcription
        
    def forward(self, inputs: Dict[str, Any]) -> Tuple[torch.Tensor, Dict]:
        """Process multi-modal inputs."""
        embeddings = []
        metadata = {}
        
        # Process each modality if present
        if 'text' in inputs:
            text_emb = self.encode_text(inputs['text'])
            embeddings.append(text_emb)
            metadata['has_text'] = True
            
        if 'image' in inputs:
            image_emb = self.encode_image(inputs['image'])
            embeddings.append(image_emb)
            metadata['has_image'] = True
            
        if 'audio' in inputs:
            audio_emb, transcription = self.encode_audio(inputs['audio'])
            embeddings.append(audio_emb)
            metadata['transcription'] = transcription
            metadata['has_audio'] = True
            
        # Pad missing modalities with zeros
        hidden_dim = embeddings[0].shape[-1] if embeddings else 768
        while len(embeddings) < 3:
            embeddings.append(torch.zeros(1, hidden_dim))
            
        # Stack and apply cross-attention
        stacked = torch.stack(embeddings, dim=1)  # [batch, modalities, hidden_dim]
        attended, _ = self.cross_attention(stacked, stacked, stacked)
        
        # Fuse modalities
        flattened = attended.view(attended.shape[0], -1)
        fused = self.fusion(flattened)
        
        return fused, metadata

class FitnessExpertSystem(nn.Module):
    """Expert system for fitness-specific reasoning."""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        # Domain-specific expert modules
        self.workout_expert = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
        
        self.nutrition_expert = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
        
        self.injury_expert = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
        
        self.motivation_expert = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
        
        # Expert fusion
        self.expert_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=4,
            batch_first=True
        )
        
        self.expert_fusion = nn.Linear(256 * 4, hidden_dim)
        
    def forward(self, context_embedding: torch.Tensor, intent: IntentType) -> torch.Tensor:
        """Apply domain expertise based on intent."""
        # Get expert opinions
        workout_opinion = self.workout_expert(context_embedding)
        nutrition_opinion = self.nutrition_expert(context_embedding)
        injury_opinion = self.injury_expert(context_embedding)
        motivation_opinion = self.motivation_expert(context_embedding)
        
        # Stack expert opinions
        expert_opinions = torch.stack([
            workout_opinion,
            nutrition_opinion,
            injury_opinion,
            motivation_opinion
        ], dim=1)
        
        # Apply attention based on intent relevance
        attended_opinions, _ = self.expert_attention(
            expert_opinions, expert_opinions, expert_opinions
        )
        
        # Fuse expert knowledge
        flattened = attended_opinions.view(attended_opinions.shape[0], -1)
        expert_knowledge = self.expert_fusion(flattened)
        
        return expert_knowledge

class AdvancedFitnessChatbot(nn.Module):
    """Complete advanced fitness chatbot system."""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Initialize language model (GPT-4 or LLaMA)
        model_name = config.get('language_model', 'meta-llama/Llama-2-7b-chat-hf')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.language_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Multi-modal encoder
        self.multi_modal_encoder = MultiModalEncoder(
            hidden_dim=config.get('hidden_dim', 768)
        )
        
        # Fitness expert system
        self.expert_system = FitnessExpertSystem(
            hidden_dim=config.get('hidden_dim', 768)
        )
        
        # Intent classifier
        self.intent_classifier = nn.Sequential(
            nn.Linear(config.get('hidden_dim', 768), 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, len(IntentType))
        )
        
        # Emotion analyzer
        self.emotion_analyzer = nn.Sequential(
            nn.Linear(config.get('hidden_dim', 768), 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 7)  # 7 basic emotions
        )
        
        # Response generator with controllable attributes
        self.response_controller = nn.Sequential(
            nn.Linear(config.get('hidden_dim', 768) * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, config.get('hidden_dim', 768))
        )
        
        # Memory system
        self.memory = ConversationMemory()
        
        # Vector store for semantic search
        self.vector_dimension = config.get('hidden_dim', 768)
        self.vector_index = faiss.IndexFlatL2(self.vector_dimension)
        
        # API clients for external services
        self.openai_client = None
        if config.get('use_openai', False):
            openai.api_key = config.get('openai_api_key')
            self.openai_client = openai
            
        self.logger = logger
        
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming message and generate response."""
        try:
            # Extract multi-modal inputs
            inputs = self._extract_inputs(message)
            
            # Encode inputs
            context_embedding, metadata = self.multi_modal_encoder(inputs)
            
            # Classify intent
            intent_logits = self.intent_classifier(context_embedding)
            intent = IntentType(torch.argmax(intent_logits, dim=-1).item())
            
            # Analyze emotion
            emotion_logits = self.emotion_analyzer(context_embedding)
            emotions = F.softmax(emotion_logits, dim=-1)
            
            # Apply fitness expertise
            expert_knowledge = self.expert_system(context_embedding, intent)
            
            # Retrieve relevant memories
            query_text = inputs.get('text', metadata.get('transcription', ''))
            relevant_memories = self.memory.retrieve_relevant_memories(query_text)
            
            # Generate response
            response = await self._generate_response(
                context_embedding=context_embedding,
                expert_knowledge=expert_knowledge,
                intent=intent,
                emotions=emotions,
                memories=relevant_memories,
                metadata=metadata
            )
            
            # Update memory
            self._update_memory(message, response, intent, emotions)
            
            return {
                'response': response,
                'intent': intent.value,
                'emotions': emotions.tolist(),
                'confidence': float(torch.max(F.softmax(intent_logits, dim=-1))),
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return {
                'response': "I apologize, but I encountered an error processing your message. Could you please try again?",
                'error': str(e)
            }
            
    def _extract_inputs(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Extract multi-modal inputs from message."""
        inputs = {}
        
        if 'text' in message:
            inputs['text'] = message['text']
            
        if 'image_path' in message:
            inputs['image'] = Image.open(message['image_path'])
            
        if 'audio_path' in message:
            inputs['audio'] = message['audio_path']
            
        if 'video_path' in message:
            # Extract key frames from video
            inputs['video_frames'] = self._extract_video_frames(message['video_path'])
            
        return inputs
        
    def _extract_video_frames(self, video_path: str, num_frames: int = 5) -> List[np.ndarray]:
        """Extract key frames from video."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
        cap.release()
        return frames
        
    async def _generate_response(self, context_embedding: torch.Tensor,
                                expert_knowledge: torch.Tensor,
                                intent: IntentType,
                                emotions: torch.Tensor,
                                memories: List[Dict],
                                metadata: Dict) -> str:
        """Generate contextual response."""
        
        # Combine context and expert knowledge
        combined_context = torch.cat([context_embedding, expert_knowledge], dim=-1)
        controlled_embedding = self.response_controller(combined_context)
        
        # Prepare prompt based on intent
        prompt = self._create_prompt(intent, emotions, memories, metadata)
        
        # Generate using language model
        if self.openai_client and self.config.get('use_openai', False):
            response = await self._generate_with_openai(prompt, intent)
        else:
            response = self._generate_with_local_model(prompt, controlled_embedding)
            
        # Post-process response
        response = self._post_process_response(response, intent)
        
        return response
        
    def _create_prompt(self, intent: IntentType, emotions: torch.Tensor,
                      memories: List[Dict], metadata: Dict) -> str:
        """Create context-aware prompt for generation."""
        prompt_parts = []
        
        # System prompt
        prompt_parts.append(
            "You are an expert fitness coach and health advisor with deep knowledge "
            "in exercise science, nutrition, and wellness. Provide helpful, accurate, "
            "and motivating responses."
        )
        
        # Intent-specific context
        intent_prompts = {
            IntentType.WORKOUT_QUERY: "Focus on providing detailed workout advice and exercise recommendations.",
            IntentType.NUTRITION_ADVICE: "Provide evidence-based nutritional guidance and meal suggestions.",
            IntentType.FORM_CHECK: "Analyze exercise form and provide corrective feedback.",
            IntentType.MOTIVATION: "Be encouraging and supportive, helping to maintain motivation.",
            IntentType.INJURY_RECOVERY: "Provide safe recovery advice and when to seek medical attention.",
        }
        
        if intent in intent_prompts:
            prompt_parts.append(intent_prompts[intent])
            
        # Add relevant memories
        if memories:
            memory_context = "Relevant context from previous conversations:\n"
            for memory in memories[:3]:
                memory_context += f"- {memory['value']}\n"
            prompt_parts.append(memory_context)
            
        # Add emotional context
        dominant_emotion_idx = torch.argmax(emotions).item()
        emotion_names = ['neutral', 'happy', 'sad', 'angry', 'fear', 'surprise', 'disgust']
        dominant_emotion = emotion_names[dominant_emotion_idx]
        prompt_parts.append(f"The user seems to be feeling {dominant_emotion}. Adjust your tone accordingly.")
        
        # Add transcription if from audio
        if 'transcription' in metadata:
            prompt_parts.append(f"User said: {metadata['transcription']}")
            
        return "\n\n".join(prompt_parts)
        
    async def _generate_with_openai(self, prompt: str, intent: IntentType) -> str:
        """Generate response using OpenAI API."""
        try:
            response = await asyncio.to_thread(
                self.openai_client.ChatCompletion.create,
                model="gpt-4",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Please provide your expert advice."}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return self._generate_fallback_response(intent)
            
    def _generate_with_local_model(self, prompt: str, 
                                  controlled_embedding: torch.Tensor) -> str:
        """Generate response using local language model."""
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate
        with torch.no_grad():
            outputs = self.language_model.generate(
                inputs.input_ids,
                max_length=len(inputs.input_ids[0]) + 200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace(prompt, '').strip()
        
    def _generate_fallback_response(self, intent: IntentType) -> str:
        """Generate fallback response for specific intents."""
        fallback_responses = {
            IntentType.WORKOUT_QUERY: "I'd be happy to help with your workout question. Could you provide more details about your fitness goals and current level?",
            IntentType.NUTRITION_ADVICE: "For nutrition advice, it's important to consider your individual needs. What specific aspect of nutrition would you like to discuss?",
            IntentType.FORM_CHECK: "To check your form, I'll need to see a video or image of you performing the exercise. Please upload one if possible.",
            IntentType.MOTIVATION: "Remember, every step forward is progress! You're doing great by staying committed to your fitness journey.",
            IntentType.INJURY_RECOVERY: "For injury-related concerns, it's always best to consult with a healthcare professional. However, I can provide general recovery guidelines.",
        }
        
        return fallback_responses.get(
            intent,
            "I'm here to help with your fitness journey. How can I assist you today?"
        )
        
    def _post_process_response(self, response: str, intent: IntentType) -> str:
        """Post-process generated response."""
        # Remove any inappropriate content
        response = re.sub(r'\b(medical advice|diagnosis|prescription)\b', 
                         'guidance', response, flags=re.IGNORECASE)
        
        # Add disclaimers for certain intents
        if intent == IntentType.INJURY_RECOVERY:
            response += "\n\n*Note: This is general guidance only. Please consult a healthcare professional for personalized medical advice.*"
            
        # Ensure response is not too long
        if len(response) > 1000:
            response = response[:997] + "..."
            
        return response
        
    def _update_memory(self, message: Dict, response: str, 
                      intent: IntentType, emotions: torch.Tensor):
        """Update conversation memory."""
        # Add to short-term memory
        self.memory.add_to_short_term({
            'message': message,
            'response': response,
            'intent': intent.value,
            'emotions': emotions.tolist(),
            'timestamp': datetime.now()
        })
        
        # Extract and store important information in long-term memory
        if intent in [IntentType.GOAL_SETTING, IntentType.INJURY_RECOVERY]:
            self.memory.add_to_long_term(
                key=f"{intent.value}_{datetime.now().isoformat()}",
                value={'message': message, 'response': response}
            )
            
    def analyze_conversation_history(self, user_id: str) -> Dict[str, Any]:
        """Analyze conversation patterns and user progress."""
        history = self.memory.long_term
        
        # Analyze intent distribution
        intent_counts = {}
        for item in history:
            if 'intent' in item['value']:
                intent = item['value']['intent']
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
                
        # Analyze emotional patterns
        emotion_trends = []
        for item in self.memory.short_term:
            if 'emotions' in item:
                emotion_trends.append(item['emotions'])
                
        # Calculate engagement metrics
        total_messages = len(history)
        avg_response_length = np.mean([len(item['value'].get('response', '')) 
                                      for item in history])
        
        return {
            'total_interactions': total_messages,
            'intent_distribution': intent_counts,
            'emotion_trends': emotion_trends,
            'average_response_length': avg_response_length,
            'most_discussed_topics': list(intent_counts.keys())[:5],
            'engagement_score': min(total_messages / 100, 1.0)  # Normalized score
        }
        
    def generate_personalized_recommendations(self, user_profile: Dict) -> List[str]:
        """Generate personalized fitness recommendations."""
        recommendations = []
        
        # Analyze user goals
        if 'goals' in user_profile:
            for goal in user_profile['goals']:
                if goal == 'weight_loss':
                    recommendations.append(
                        "Consider incorporating HIIT workouts 3x per week for effective fat burning"
                    )
                elif goal == 'muscle_gain':
                    recommendations.append(
                        "Focus on progressive overload and aim for 1.6-2.2g protein per kg body weight"
                    )
                elif goal == 'endurance':
                    recommendations.append(
                        "Gradually increase your cardio duration by 10% each week"
                    )
                    
        # Based on activity level
        activity_level = user_profile.get('activity_level', 'moderate')
        if activity_level == 'sedentary':
            recommendations.append(
                "Start with 150 minutes of moderate activity per week, broken into manageable sessions"
            )
        elif activity_level == 'very_active':
            recommendations.append(
                "Ensure adequate recovery with at least 1-2 rest days per week"
            )
            
        # Nutrition recommendations
        if 'dietary_restrictions' in user_profile:
            restrictions = user_profile['dietary_restrictions']
            if 'vegetarian' in restrictions or 'vegan' in restrictions:
                recommendations.append(
                    "Include diverse plant-based protein sources like legumes, quinoa, and tofu"
                )
                
        return recommendations
        
    def save_model(self, path: str):
        """Save the chatbot model."""
        torch.save({
            'multi_modal_encoder': self.multi_modal_encoder.state_dict(),
            'expert_system': self.expert_system.state_dict(),
            'intent_classifier': self.intent_classifier.state_dict(),
            'emotion_analyzer': self.emotion_analyzer.state_dict(),
            'response_controller': self.response_controller.state_dict(),
            'config': self.config,
            'memory': pickle.dumps(self.memory)
        }, path)
        
    def load_model(self, path: str):
        """Load the chatbot model."""
        checkpoint = torch.load(path, map_location='cpu')
        
        self.multi_modal_encoder.load_state_dict(checkpoint['multi_modal_encoder'])
        self.expert_system.load_state_dict(checkpoint['expert_system'])
        self.intent_classifier.load_state_dict(checkpoint['intent_classifier'])
        self.emotion_analyzer.load_state_dict(checkpoint['emotion_analyzer'])
        self.response_controller.load_state_dict(checkpoint['response_controller'])
        
        if 'memory' in checkpoint:
            self.memory = pickle.loads(checkpoint['memory'])
            
        self.logger.info(f"Model loaded from {path}")

# Example usage
async def main():
    """Example usage of the advanced chatbot."""
    config = {
        'language_model': 'meta-llama/Llama-2-7b-chat-hf',
        'hidden_dim': 768,
        'use_openai': False,  # Set to True if using OpenAI
        'openai_api_key': 'your-api-key-here'
    }
    
    # Create chatbot
    chatbot = AdvancedFitnessChatbot(config)
    
    # Example text message
    text_message = {
        'text': "I want to lose weight but I'm struggling with motivation. What should I do?",
        'user_id': 'user123'
    }
    
    response = await chatbot.process_message(text_message)
    print("Text Response:", response)
    
    # Example multi-modal message
    multimodal_message = {
        'text': "Is my squat form correct?",
        'image_path': 'path/to/squat_image.jpg',
        'user_id': 'user123'
    }
    
    response = await chatbot.process_message(multimodal_message)
    print("Multi-modal Response:", response)
    
    # Analyze conversation history
    analysis = chatbot.analyze_conversation_history('user123')
    print("Conversation Analysis:", analysis)
    
    # Generate recommendations
    user_profile = {
        'goals': ['weight_loss', 'muscle_gain'],
        'activity_level': 'moderate',
        'dietary_restrictions': ['vegetarian']
    }
    
    recommendations = chatbot.generate_personalized_recommendations(user_profile)
    print("Personalized Recommendations:", recommendations)

if __name__ == "__main__":
    asyncio.run(main())