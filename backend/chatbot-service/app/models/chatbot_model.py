"""
Chatbot Model - Core NLP and conversation logic
Uses DistilBERT for intent classification and GPT-2 for response generation
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    pipeline
)
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class ChatbotModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Model paths
        self.intent_model_path = os.getenv("INTENT_MODEL_PATH", "distilbert-base-uncased")
        self.response_model_path = os.getenv("RESPONSE_MODEL_PATH", "gpt2")
        
        # Intent categories
        self.intent_labels = [
            "fitness_question",
            "diet_question",
            "workout_request",
            "motivation",
            "progress_check",
            "form_check",
            "general_chat",
            "greeting",
            "goodbye"
        ]
        
        # Entity types
        self.entity_types = [
            "exercise",
            "food",
            "body_part",
            "time",
            "number",
            "goal"
        ]
        
        self.models_loaded = False
    
    async def initialize(self):
        """Initialize all models and components"""
        try:
            logger.info("Initializing chatbot models...")
            
            # Load intent classifier
            self.intent_tokenizer = AutoTokenizer.from_pretrained(self.intent_model_path)
            self.intent_model = AutoModelForSequenceClassification.from_pretrained(
                self.intent_model_path,
                num_labels=len(self.intent_labels)
            ).to(self.device)
            
            # Load response generator
            self.response_tokenizer = GPT2Tokenizer.from_pretrained(self.response_model_path)
            self.response_tokenizer.pad_token = self.response_tokenizer.eos_token
            self.response_model = GPT2LMHeadModel.from_pretrained(self.response_model_path).to(self.device)
            
            # Load sentence embedder for context understanding
            self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load NER pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Load fitness knowledge base
            self.load_knowledge_base()
            
            self.models_loaded = True
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
    
    def load_knowledge_base(self):
        """Load fitness knowledge base and FAQ responses"""
        self.knowledge_base = {
            "calorie_info": {
                "rice": "130 calories per 100g cooked",
                "chicken": "165 calories per 100g",
                "banana": "89 calories per 100g",
                "apple": "52 calories per 100g",
                "eggs": "155 calories per 100g"
            },
            "exercise_info": {
                "pushup": "Works chest, shoulders, triceps. Great for upper body strength.",
                "squat": "Works quads, glutes, hamstrings. Essential for leg strength.",
                "plank": "Core stability exercise. Hold for 30-60 seconds.",
                "deadlift": "Full body exercise. Focus on form to prevent injury."
            },
            "motivation_quotes": [
                "Every workout counts! You're one step closer to your goal! 💪",
                "Consistency is key! Keep pushing forward! 🚀",
                "Your body can stand almost anything. It's your mind you have to convince! 🧠",
                "The pain you feel today will be the strength you feel tomorrow! 💯",
                "You're stronger than you think! Keep going! 🔥"
            ],
            "tips": {
                "weight_loss": "Create a 500 calorie deficit daily for 1lb/week loss. Combine cardio with strength training.",
                "muscle_gain": "Eat protein (0.8-1g per lb body weight), progressive overload, and rest adequately.",
                "hydration": "Drink at least 8 glasses of water daily, more during workouts.",
                "sleep": "Aim for 7-9 hours of quality sleep for optimal recovery."
            }
        }
    
    async def process_message(
        self, 
        user_id: str, 
        message: str, 
        context: Optional[Dict] = None
    ) -> Dict:
        """Process user message and generate response"""
        
        if not self.models_loaded:
            await self.initialize()
        
        # Clean and prepare message
        message = message.strip().lower()
        
        # Classify intent
        intent, intent_confidence = await self.classify_intent(message)
        
        # Extract entities
        entities = await self.extract_entities(message)
        
        # Generate response based on intent
        response = await self.generate_response(
            message=message,
            intent=intent,
            entities=entities,
            context=context
        )
        
        # Get follow-up suggestions
        suggestions = self.get_suggestions(intent, entities)
        
        return {
            "response": response,
            "intent": intent,
            "intent_confidence": intent_confidence,
            "entities": entities,
            "suggestions": suggestions,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def classify_intent(self, message: str) -> Tuple[str, float]:
        """Classify the intent of the message"""
        
        # Rule-based classification for common patterns
        if any(word in message for word in ["hello", "hi", "hey", "greetings"]):
            return "greeting", 0.95
        
        if any(word in message for word in ["bye", "goodbye", "see you", "later"]):
            return "goodbye", 0.95
        
        if any(word in message for word in ["calorie", "nutrition", "diet", "food", "meal"]):
            return "diet_question", 0.85
        
        if any(word in message for word in ["workout", "exercise", "training", "routine"]):
            return "workout_request", 0.85
        
        if any(word in message for word in ["motivate", "tired", "can't", "hard", "difficult"]):
            return "motivation", 0.80
        
        if any(word in message for word in ["form", "posture", "correct", "right way"]):
            return "form_check", 0.80
        
        if any(word in message for word in ["progress", "results", "improvement", "gains"]):
            return "progress_check", 0.80
        
        # Use BERT model for complex classification
        inputs = self.intent_tokenizer(
            message,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.intent_model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Default to fitness_question if confidence is low
        if confidence < 0.5:
            return "fitness_question", 0.5
        
        return self.intent_labels[predicted_class], confidence
    
    async def extract_entities(self, message: str) -> List[Dict]:
        """Extract entities from the message"""
        entities = []
        
        # Use NER pipeline
        ner_results = self.ner_pipeline(message)
        
        for entity in ner_results:
            entities.append({
                "text": entity["word"],
                "type": entity["entity"],
                "score": entity["score"]
            })
        
        # Custom entity extraction for fitness domain
        # Exercise names
        exercises = ["pushup", "squat", "deadlift", "plank", "bench press", "curl", "row"]
        for exercise in exercises:
            if exercise in message:
                entities.append({
                    "text": exercise,
                    "type": "exercise",
                    "score": 0.9
                })
        
        # Food items
        foods = ["rice", "chicken", "banana", "apple", "eggs", "protein", "carbs"]
        for food in foods:
            if food in message:
                entities.append({
                    "text": food,
                    "type": "food",
                    "score": 0.9
                })
        
        # Body parts
        body_parts = ["chest", "legs", "arms", "back", "shoulders", "abs", "core"]
        for part in body_parts:
            if part in message:
                entities.append({
                    "text": part,
                    "type": "body_part",
                    "score": 0.85
                })
        
        return entities
    
    async def generate_response(
        self,
        message: str,
        intent: str,
        entities: List[Dict],
        context: Optional[Dict] = None
    ) -> str:
        """Generate contextual response based on intent and entities"""
        
        # Handle different intents
        if intent == "greeting":
            return "Hey there! 👋 Ready to crush your fitness goals today? How can I help you?"
        
        elif intent == "goodbye":
            return "Great work today! Remember to stay hydrated and get good rest. See you next time! 💪"
        
        elif intent == "diet_question":
            return await self.handle_diet_question(message, entities)
        
        elif intent == "workout_request":
            return await self.handle_workout_request(message, entities)
        
        elif intent == "motivation":
            import random
            return random.choice(self.knowledge_base["motivation_quotes"])
        
        elif intent == "form_check":
            return await self.handle_form_check(message, entities)
        
        elif intent == "progress_check":
            return await self.handle_progress_check(context)
        
        elif intent == "fitness_question":
            return await self.handle_fitness_question(message, entities)
        
        else:
            # Generate response using GPT-2
            return await self.generate_gpt2_response(message, context)
    
    async def handle_diet_question(self, message: str, entities: List[Dict]) -> str:
        """Handle diet and nutrition questions"""
        
        # Check for calorie queries
        food_entities = [e for e in entities if e["type"] == "food"]
        
        if "calorie" in message and food_entities:
            food = food_entities[0]["text"]
            if food in self.knowledge_base["calorie_info"]:
                calorie_info = self.knowledge_base["calorie_info"][food]
                return f"{food.capitalize()} contains {calorie_info}. Need help planning your meals? 🥗"
        
        if "weight loss" in message or "lose weight" in message:
            return self.knowledge_base["tips"]["weight_loss"] + " Want me to create a personalized meal plan?"
        
        if "muscle" in message or "gain" in message:
            return self.knowledge_base["tips"]["muscle_gain"] + " Should I suggest high-protein meal options?"
        
        # Default diet response
        return "For a balanced diet, aim for 40% carbs, 30% protein, and 30% healthy fats. What specific nutrition goal do you have?"
    
    async def handle_workout_request(self, message: str, entities: List[Dict]) -> str:
        """Handle workout and exercise requests"""
        
        exercise_entities = [e for e in entities if e["type"] == "exercise"]
        body_part_entities = [e for e in entities if e["type"] == "body_part"]
        
        if exercise_entities:
            exercise = exercise_entities[0]["text"]
            if exercise in self.knowledge_base["exercise_info"]:
                return self.knowledge_base["exercise_info"][exercise]
        
        if body_part_entities:
            body_part = body_part_entities[0]["text"]
            return f"Great choice! For {body_part}, I recommend 3-4 exercises, 3 sets of 8-12 reps. Want a detailed {body_part} workout plan?"
        
        if "beginner" in message:
            return "Perfect! Start with: 3x10 pushups, 3x15 squats, 3x30s plank, 2x10 lunges each leg. Rest 60s between sets. How does that sound?"
        
        # Default workout response
        return "I'll create a workout based on your fitness level. Are you beginner, intermediate, or advanced? What's your main goal?"
    
    async def handle_form_check(self, message: str, entities: List[Dict]) -> str:
        """Handle form and technique questions"""
        
        exercise_entities = [e for e in entities if e["type"] == "exercise"]
        
        if exercise_entities:
            exercise = exercise_entities[0]["text"]
            tips = {
                "squat": "Keep chest up, knees tracking over toes, weight on heels. Go down until thighs are parallel to ground.",
                "pushup": "Keep body straight, hands shoulder-width apart, lower until chest nearly touches ground.",
                "deadlift": "Keep back straight, chest up, drive through heels, bar close to body throughout movement.",
                "plank": "Keep body straight from head to heels, engage core, breathe normally."
            }
            
            if exercise in tips:
                return f"Form tips for {exercise}: {tips[exercise]} Want me to analyze a video of your form?"
        
        return "Good form is crucial! Upload a video or describe the exercise, and I'll guide you through proper technique. Safety first! 🎯"
    
    async def handle_progress_check(self, context: Optional[Dict]) -> str:
        """Handle progress tracking questions"""
        
        if context and "last_workout" in context:
            days_since = (datetime.utcnow() - context["last_workout"]).days
            if days_since > 3:
                return f"It's been {days_since} days since your last workout. Time to get back on track! Your consistency affects your results. Ready for today's session?"
        
        return "Track your progress weekly! Measure weight, body measurements, and strength gains. Want me to set up a progress tracking plan? 📊"
    
    async def handle_fitness_question(self, message: str, entities: List[Dict]) -> str:
        """Handle general fitness questions"""
        
        if "hydration" in message or "water" in message:
            return self.knowledge_base["tips"]["hydration"] + " I can remind you to drink water throughout the day!"
        
        if "sleep" in message or "rest" in message or "recovery" in message:
            return self.knowledge_base["tips"]["sleep"] + " Recovery is when muscles grow!"
        
        # Generate contextual response
        return await self.generate_gpt2_response(f"Fitness advice for: {message}")
    
    async def generate_gpt2_response(self, message: str, context: Optional[Dict] = None) -> str:
        """Generate response using GPT-2 model"""
        
        # Prepare prompt with context
        prompt = "You are a friendly fitness coach. "
        if context:
            prompt += f"Context: User's goal is {context.get('goal', 'general fitness')}. "
        prompt += f"User asks: {message}\nCoach response:"
        
        # Tokenize input
        inputs = self.response_tokenizer.encode(
            prompt,
            return_tensors="pt",
            max_length=150,
            truncation=True
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.response_model.generate(
                inputs,
                max_length=200,
                num_return_sequences=1,
                temperature=0.8,
                pad_token_id=self.response_tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.9
            )
        
        # Decode response
        response = self.response_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        response = response.split("Coach response:")[-1].strip()
        
        # Limit response length for conversational feel
        sentences = response.split(".")
        if len(sentences) > 2:
            response = ". ".join(sentences[:2]) + "."
        
        return response if response else "I'm here to help with your fitness journey! Could you tell me more about what you need?"
    
    def get_suggestions(self, intent: str, entities: List[Dict]) -> List[str]:
        """Get follow-up suggestions based on current conversation"""
        
        suggestions_map = {
            "greeting": [
                "Show me today's workout",
                "Check my progress",
                "I need diet advice"
            ],
            "diet_question": [
                "Create a meal plan",
                "Calculate my daily calories",
                "High protein foods"
            ],
            "workout_request": [
                "Show exercise form",
                "Create weekly routine",
                "Rest day activities"
            ],
            "motivation": [
                "Set a new goal",
                "Show my achievements",
                "Success stories"
            ],
            "form_check": [
                "Common mistakes",
                "Upload form video",
                "Alternative exercises"
            ],
            "progress_check": [
                "Update measurements",
                "Set new goals",
                "Compare with last month"
            ]
        }
        
        return suggestions_map.get(intent, [
            "Ask about workouts",
            "Get diet advice",
            "Check your progress"
        ])