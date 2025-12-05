"""
Advanced NLP Fitness Chatbot Model
Conversational AI for health & fitness with intent classification, entity extraction, and context management
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DistilBertModel, GPT2Model, GPT2Tokenizer,
    pipeline, BertForTokenClassification
)
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import re
from dataclasses import dataclass
from enum import Enum
import spacy
from datetime import datetime, timedelta
import random

# Intent categories
class IntentCategory(Enum):
    FITNESS_QUESTION = "fitness_question"
    DIET_QUESTION = "diet_question"
    MOTIVATION = "motivation"
    WORKOUT_REQUEST = "workout_request"
    PROGRESS_CHECK = "progress_check"
    REMINDER_SET = "reminder_set"
    GENERAL_CHAT = "general_chat"
    MEDICAL_QUERY = "medical_query"

# Entity types
class EntityType(Enum):
    EXERCISE = "exercise"
    FOOD = "food"
    TIME = "time"
    BODY_PART = "body_part"
    METRIC = "metric"
    GOAL = "goal"
    DURATION = "duration"

@dataclass
class ConversationContext:
    """Stores conversation context across sessions"""
    user_id: str
    history: List[Dict[str, str]]
    user_profile: Dict[str, Any]
    current_goal: Optional[str]
    last_interaction: datetime
    session_count: int
    
class MultiHeadAttentionLayer(nn.Module):
    """Custom multi-head attention for context understanding"""
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = x.shape[:2]
        
        # Linear transformations and reshape
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        
        # Reshape and output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.output(context)
        
        return output

class IntentClassifier(nn.Module):
    """Deep learning model for intent classification"""
    def __init__(self, input_dim: int = 768, num_intents: int = 8):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Freeze BERT layers initially
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Classification layers
        self.attention = MultiHeadAttentionLayer(input_dim)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_intents)
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Apply attention
        attended = self.attention(sequence_output, attention_mask.unsqueeze(1).unsqueeze(2))
        
        # Pool the outputs
        pooled = attended.mean(dim=1)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits

class EntityExtractor(nn.Module):
    """BiLSTM-CRF model for entity extraction"""
    def __init__(self, vocab_size: int, embedding_dim: int = 100, 
                 hidden_dim: int = 256, num_entities: int = 7):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, 
                           num_layers=2, bidirectional=True, 
                           batch_first=True, dropout=0.3)
        self.hidden2tag = nn.Linear(hidden_dim, num_entities)
        self.crf = CRFLayer(num_entities)
        
    def forward(self, sentences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(sentences)
        packed = nn.utils.rnn.pack_padded_sequence(embeds, lengths, 
                                                   batch_first=True, 
                                                   enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        emissions = self.hidden2tag(unpacked)
        return emissions

class CRFLayer(nn.Module):
    """Conditional Random Field layer for sequence labeling"""
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        
    def forward(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute the conditional log likelihood"""
        return self._compute_log_likelihood(emissions, mask)
    
    def _compute_log_likelihood(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Simplified CRF computation
        batch_size, seq_len, num_tags = emissions.shape
        
        # Forward algorithm
        alpha = self.start_transitions + emissions[:, 0]
        
        for t in range(1, seq_len):
            emit_score = emissions[:, t].unsqueeze(1)
            trans_score = self.transitions.unsqueeze(0)
            next_alpha = alpha.unsqueeze(2) + emit_score + trans_score
            alpha = torch.logsumexp(next_alpha, dim=1)
            alpha = torch.where(mask[:, t].unsqueeze(1), alpha, alpha)
        
        # Add end transitions
        alpha = alpha + self.end_transitions
        
        return torch.logsumexp(alpha, dim=1)

class ResponseGenerator(nn.Module):
    """GPT-2 based response generator with fine-tuning for fitness domain"""
    def __init__(self, model_name: str = 'gpt2'):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Fine-tuning layers
        hidden_size = self.gpt2.config.hidden_size
        self.domain_adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.lm_head = nn.Linear(hidden_size, self.gpt2.config.vocab_size)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Apply domain adaptation
        adapted = self.domain_adapter(hidden_states)
        
        # Generate logits
        logits = self.lm_head(adapted)
        
        return logits
    
    def generate_response(self, prompt: str, max_length: int = 100, 
                         temperature: float = 0.7) -> str:
        """Generate response using nucleus sampling"""
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.gpt2.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

class FitnessChatbot:
    """Main chatbot system integrating all components"""
    
    def __init__(self, model_path: Optional[str] = None):
        # Initialize models
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor(vocab_size=30000)
        self.response_generator = ResponseGenerator()
        
        # Initialize NLP tools
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize tokenizers
        self.bert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Context management
        self.contexts: Dict[str, ConversationContext] = {}
        
        # Knowledge base
        self.knowledge_base = self._load_knowledge_base()
        
        # Response templates
        self.response_templates = self._load_response_templates()
        
        if model_path:
            self.load_model(model_path)
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load fitness and nutrition knowledge base"""
        return {
            "exercises": {
                "push_up": {"muscles": ["chest", "triceps", "shoulders"], 
                           "difficulty": "beginner", "equipment": "none"},
                "squat": {"muscles": ["quadriceps", "glutes", "hamstrings"], 
                         "difficulty": "beginner", "equipment": "none"},
                "deadlift": {"muscles": ["back", "glutes", "hamstrings"], 
                            "difficulty": "intermediate", "equipment": "barbell"},
                "plank": {"muscles": ["core", "shoulders"], 
                         "difficulty": "beginner", "equipment": "none"},
                "pull_up": {"muscles": ["back", "biceps"], 
                           "difficulty": "intermediate", "equipment": "pull_up_bar"}
            },
            "nutrition": {
                "rice": {"calories": 130, "carbs": 28, "protein": 2.7, "fat": 0.3},
                "chicken": {"calories": 165, "carbs": 0, "protein": 31, "fat": 3.6},
                "broccoli": {"calories": 31, "carbs": 6, "protein": 2.5, "fat": 0.4},
                "banana": {"calories": 105, "carbs": 27, "protein": 1.3, "fat": 0.4},
                "eggs": {"calories": 155, "carbs": 1.1, "protein": 13, "fat": 11}
            },
            "tips": {
                "weight_loss": [
                    "Aim for a 500-calorie deficit per day for healthy weight loss",
                    "Combine cardio with strength training for optimal results",
                    "Stay hydrated - drink at least 8 glasses of water daily",
                    "Get 7-9 hours of quality sleep each night"
                ],
                "muscle_gain": [
                    "Consume 1.6-2.2g of protein per kg of body weight",
                    "Progressive overload is key - gradually increase weights",
                    "Allow 48 hours rest between training same muscle groups",
                    "Eat in a slight caloric surplus (300-500 calories)"
                ]
            }
        }
    
    def _load_response_templates(self) -> Dict[str, List[str]]:
        """Load response templates for different intents"""
        return {
            IntentCategory.FITNESS_QUESTION.value: [
                "Great question about fitness! {answer}",
                "Here's what you need to know: {answer}",
                "Based on fitness science: {answer}"
            ],
            IntentCategory.DIET_QUESTION.value: [
                "Regarding nutrition: {answer}",
                "Here's the nutritional info: {answer}",
                "For your diet question: {answer}"
            ],
            IntentCategory.MOTIVATION.value: [
                "You've got this! {message}",
                "Stay strong! {message}",
                "Remember why you started! {message}"
            ],
            IntentCategory.WORKOUT_REQUEST.value: [
                "Here's your workout: {workout}",
                "I've prepared this routine for you: {workout}",
                "Let's get moving with: {workout}"
            ],
            IntentCategory.MEDICAL_QUERY.value: [
                "I can only provide general fitness guidance. For medical advice, please consult a healthcare professional.",
                "This sounds like a medical concern. It's best to speak with your doctor.",
                "While I can help with fitness, medical questions should be directed to a qualified healthcare provider."
            ]
        }
    
    def process_query(self, user_id: str, query: str) -> Dict[str, Any]:
        """Main entry point for processing user queries"""
        
        # Get or create context
        context = self._get_or_create_context(user_id)
        
        # Classify intent
        intent = self._classify_intent(query)
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Check for medical queries first
        if intent == IntentCategory.MEDICAL_QUERY:
            response = self._get_medical_disclaimer()
        else:
            # Generate response based on intent and entities
            response = self._generate_response(intent, entities, context, query)
        
        # Update context
        self._update_context(context, query, response)
        
        # Format output
        output = {
            "response": response,
            "intent": intent.value,
            "entities": entities,
            "context_aware": True,
            "timestamp": datetime.now().isoformat()
        }
        
        return output
    
    def _classify_intent(self, query: str) -> IntentCategory:
        """Classify the intent of the user query"""
        
        # Tokenize input
        inputs = self.bert_tokenizer(query, return_tensors='pt', 
                                    padding=True, truncation=True, 
                                    max_length=128)
        
        # Get predictions
        with torch.no_grad():
            logits = self.intent_classifier(inputs['input_ids'], 
                                           inputs['attention_mask'])
            probs = F.softmax(logits, dim=-1)
            intent_idx = torch.argmax(probs, dim=-1).item()
        
        # Map to intent category
        intent_map = list(IntentCategory)
        return intent_map[intent_idx]
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from the query"""
        
        entities = {
            "exercises": [],
            "foods": [],
            "times": [],
            "body_parts": [],
            "metrics": [],
            "goals": []
        }
        
        # Use spaCy for NER
        doc = self.nlp(query.lower())
        
        # Extract time entities
        for ent in doc.ents:
            if ent.label_ in ["TIME", "DATE"]:
                entities["times"].append(ent.text)
        
        # Extract exercises
        for exercise in self.knowledge_base["exercises"].keys():
            if exercise in query.lower():
                entities["exercises"].append(exercise)
        
        # Extract foods
        for food in self.knowledge_base["nutrition"].keys():
            if food in query.lower():
                entities["foods"].append(food)
        
        # Extract body parts
        body_parts = ["chest", "back", "arms", "legs", "shoulders", 
                     "abs", "core", "glutes", "biceps", "triceps"]
        for part in body_parts:
            if part in query.lower():
                entities["body_parts"].append(part)
        
        # Extract metrics
        metric_patterns = [
            r'\d+\s*(?:kg|lbs?|pounds?|kilograms?)',
            r'\d+\s*(?:calories?|cals?)',
            r'\d+\s*(?:reps?|sets?)',
            r'\d+\s*(?:minutes?|mins?|hours?|hrs?)'
        ]
        for pattern in metric_patterns:
            matches = re.findall(pattern, query.lower())
            entities["metrics"].extend(matches)
        
        # Extract goals
        goal_keywords = ["lose weight", "gain muscle", "get fit", 
                        "bulk", "cut", "tone", "strength", "endurance"]
        for goal in goal_keywords:
            if goal in query.lower():
                entities["goals"].append(goal)
        
        return entities
    
    def _generate_response(self, intent: IntentCategory, entities: Dict[str, List[str]], 
                          context: ConversationContext, query: str) -> str:
        """Generate appropriate response based on intent and entities"""
        
        if intent == IntentCategory.FITNESS_QUESTION:
            return self._handle_fitness_question(entities, query)
        
        elif intent == IntentCategory.DIET_QUESTION:
            return self._handle_diet_question(entities, query)
        
        elif intent == IntentCategory.MOTIVATION:
            return self._generate_motivation(context)
        
        elif intent == IntentCategory.WORKOUT_REQUEST:
            return self._generate_workout(entities, context)
        
        elif intent == IntentCategory.PROGRESS_CHECK:
            return self._check_progress(context)
        
        elif intent == IntentCategory.REMINDER_SET:
            return self._set_reminder(entities)
        
        else:
            # Use GPT-2 for general chat
            prompt = f"Fitness coach response to: {query}\nCoach:"
            response = self.response_generator.generate_response(prompt, max_length=100)
            return response.split("Coach:")[-1].strip()
    
    def _handle_fitness_question(self, entities: Dict[str, List[str]], query: str) -> str:
        """Handle fitness-related questions"""
        
        response_parts = []
        
        # Check for exercises in query
        if entities["exercises"]:
            for exercise in entities["exercises"]:
                exercise_info = self.knowledge_base["exercises"].get(exercise, {})
                if exercise_info:
                    muscles = ", ".join(exercise_info["muscles"])
                    response_parts.append(
                        f"{exercise.replace('_', ' ').title()} targets: {muscles}. "
                        f"Difficulty: {exercise_info['difficulty']}."
                    )
        
        # Check for body parts
        if entities["body_parts"]:
            for body_part in entities["body_parts"]:
                exercises = [ex for ex, info in self.knowledge_base["exercises"].items() 
                           if body_part in info["muscles"]]
                if exercises:
                    response_parts.append(
                        f"For {body_part}, try: {', '.join(exercises[:3])}"
                    )
        
        # Check for goals
        if entities["goals"]:
            for goal in entities["goals"]:
                if "weight" in goal or "lose" in goal:
                    tips = random.choice(self.knowledge_base["tips"]["weight_loss"])
                    response_parts.append(tips)
                elif "muscle" in goal or "gain" in goal:
                    tips = random.choice(self.knowledge_base["tips"]["muscle_gain"])
                    response_parts.append(tips)
        
        if response_parts:
            return " ".join(response_parts)
        else:
            return "Could you be more specific about your fitness question? I can help with exercises, techniques, and training advice."
    
    def _handle_diet_question(self, entities: Dict[str, List[str]], query: str) -> str:
        """Handle nutrition-related questions"""
        
        response_parts = []
        
        if entities["foods"]:
            for food in entities["foods"]:
                nutrition = self.knowledge_base["nutrition"].get(food, {})
                if nutrition:
                    response_parts.append(
                        f"{food.title()} (per 100g): "
                        f"{nutrition['calories']} calories, "
                        f"{nutrition['carbs']}g carbs, "
                        f"{nutrition['protein']}g protein, "
                        f"{nutrition['fat']}g fat"
                    )
        
        if "calorie" in query.lower():
            if "deficit" in query.lower():
                response_parts.append(
                    "For weight loss, aim for a 500-calorie deficit per day. "
                    "This leads to about 1 pound of weight loss per week."
                )
            elif "surplus" in query.lower():
                response_parts.append(
                    "For muscle gain, aim for a 300-500 calorie surplus. "
                    "Combine this with strength training for best results."
                )
        
        if response_parts:
            return " ".join(response_parts)
        else:
            return "I can help with nutrition info, meal planning, and dietary advice. What would you like to know?"
    
    def _generate_motivation(self, context: ConversationContext) -> str:
        """Generate motivational messages"""
        
        motivational_quotes = [
            "Every workout counts! You're building a stronger version of yourself.",
            "Progress isn't always visible immediately, but every effort matters!",
            "Your only competition is who you were yesterday. Keep pushing!",
            "The pain you feel today will be the strength you feel tomorrow!",
            "Success is the sum of small efforts repeated day in and day out.",
            "You're not just building muscle, you're building character!",
            "Champions are made when no one is watching. Keep grinding!",
            "Your body can stand almost anything. It's your mind you have to convince."
        ]
        
        # Personalize based on context
        if context.current_goal:
            if "weight" in context.current_goal.lower():
                return f"Keep working towards your weight goal! {random.choice(motivational_quotes)}"
            elif "muscle" in context.current_goal.lower():
                return f"Building muscle takes time and consistency! {random.choice(motivational_quotes)}"
        
        return random.choice(motivational_quotes)
    
    def _generate_workout(self, entities: Dict[str, List[str]], 
                         context: ConversationContext) -> str:
        """Generate workout recommendations"""
        
        workout_parts = []
        
        # Check for specific body parts
        if entities["body_parts"]:
            for body_part in entities["body_parts"]:
                exercises = [(ex, info) for ex, info in self.knowledge_base["exercises"].items() 
                           if body_part in info["muscles"]]
                
                if exercises:
                    workout_parts.append(f"\n{body_part.upper()} WORKOUT:")
                    for ex, info in exercises[:3]:
                        exercise_name = ex.replace('_', ' ').title()
                        if info['difficulty'] == 'beginner':
                            sets_reps = "3 sets x 12-15 reps"
                        else:
                            sets_reps = "4 sets x 8-12 reps"
                        workout_parts.append(f"- {exercise_name}: {sets_reps}")
        
        # If no specific body parts, give a general workout
        if not workout_parts:
            if context.user_profile.get("fitness_level") == "beginner":
                workout_parts = [
                    "BEGINNER FULL BODY WORKOUT:",
                    "- Push-ups: 3 sets x 10-12 reps",
                    "- Bodyweight Squats: 3 sets x 15 reps",
                    "- Plank: 3 sets x 30 seconds",
                    "- Lunges: 3 sets x 10 per leg",
                    "- Mountain Climbers: 3 sets x 20 reps"
                ]
            else:
                workout_parts = [
                    "INTERMEDIATE WORKOUT:",
                    "- Push-ups: 4 sets x 15-20 reps",
                    "- Pull-ups: 4 sets x 8-12 reps",
                    "- Squats: 4 sets x 12-15 reps",
                    "- Deadlifts: 4 sets x 8-10 reps",
                    "- Plank: 3 sets x 60 seconds"
                ]
        
        workout_parts.append("\nRest 60-90 seconds between sets. Stay hydrated!")
        
        return "\n".join(workout_parts)
    
    def _check_progress(self, context: ConversationContext) -> str:
        """Check user's progress"""
        
        days_active = (datetime.now() - context.last_interaction).days
        
        if days_active == 0:
            consistency_msg = "Great to see you today!"
        elif days_active == 1:
            consistency_msg = "Welcome back! Consistency is key."
        else:
            consistency_msg = f"It's been {days_active} days. Let's get back on track!"
        
        progress_msg = f"You've had {context.session_count} training sessions. "
        
        if context.current_goal:
            progress_msg += f"Keep working towards your goal: {context.current_goal}."
        
        return f"{consistency_msg} {progress_msg}"
    
    def _set_reminder(self, entities: Dict[str, List[str]]) -> str:
        """Set workout or nutrition reminders"""
        
        if entities["times"]:
            time = entities["times"][0]
            return f"I'll remind you at {time}. Stay committed to your fitness journey!"
        else:
            return "What time would you like me to remind you? Just tell me the time and what you'd like to be reminded about."
    
    def _get_medical_disclaimer(self) -> str:
        """Return medical disclaimer"""
        templates = self.response_templates[IntentCategory.MEDICAL_QUERY.value]
        return random.choice(templates)
    
    def _get_or_create_context(self, user_id: str) -> ConversationContext:
        """Get or create user context"""
        if user_id not in self.contexts:
            self.contexts[user_id] = ConversationContext(
                user_id=user_id,
                history=[],
                user_profile={},
                current_goal=None,
                last_interaction=datetime.now(),
                session_count=0
            )
        return self.contexts[user_id]
    
    def _update_context(self, context: ConversationContext, 
                       query: str, response: str) -> None:
        """Update conversation context"""
        context.history.append({
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 10 interactions
        if len(context.history) > 10:
            context.history = context.history[-10:]
        
        context.last_interaction = datetime.now()
        context.session_count += 1
    
    def train(self, training_data: List[Dict[str, Any]], 
             epochs: int = 10, batch_size: int = 32) -> Dict[str, float]:
        """Train the chatbot models"""
        
        # Prepare data loaders
        intent_data = [(d['query'], d['intent']) for d in training_data]
        entity_data = [(d['query'], d['entities']) for d in training_data if 'entities' in d]
        
        # Training metrics
        metrics = {
            'intent_accuracy': 0.0,
            'entity_f1': 0.0,
            'response_quality': 0.0
        }
        
        # Train intent classifier
        optimizer = torch.optim.AdamW(self.intent_classifier.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for i in range(0, len(intent_data), batch_size):
                batch = intent_data[i:i+batch_size]
                queries = [q for q, _ in batch]
                intents = [IntentCategory[i].value for _, i in batch]
                
                # Tokenize
                inputs = self.bert_tokenizer(queries, return_tensors='pt', 
                                            padding=True, truncation=True, 
                                            max_length=128)
                
                # Convert intents to indices
                intent_indices = torch.tensor([list(IntentCategory).index(
                    IntentCategory(i)) for i in intents])
                
                # Forward pass
                logits = self.intent_classifier(inputs['input_ids'], 
                                               inputs['attention_mask'])
                loss = criterion(logits, intent_indices)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == intent_indices).sum().item()
                total += len(batch)
            
            metrics['intent_accuracy'] = correct / total
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, "
                  f"Accuracy: {metrics['intent_accuracy']:.4f}")
        
        return metrics
    
    def save_model(self, path: str) -> None:
        """Save model weights and configuration"""
        torch.save({
            'intent_classifier': self.intent_classifier.state_dict(),
            'entity_extractor': self.entity_extractor.state_dict(),
            'response_generator': self.response_generator.state_dict(),
            'contexts': self.contexts,
            'knowledge_base': self.knowledge_base
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model weights and configuration"""
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.intent_classifier.load_state_dict(checkpoint['intent_classifier'])
        self.entity_extractor.load_state_dict(checkpoint['entity_extractor'])
        self.response_generator.load_state_dict(checkpoint['response_generator'])
        self.contexts = checkpoint.get('contexts', {})
        self.knowledge_base = checkpoint.get('knowledge_base', self.knowledge_base)
        print(f"Model loaded from {path}")

# Example usage
if __name__ == "__main__":
    # Initialize chatbot
    chatbot = FitnessChatbot()
    
    # Example queries
    test_queries = [
        "How many calories are in 100g of rice?",
        "Show me a chest workout",
        "I want to lose weight, any tips?",
        "Remind me to drink water every 2 hours",
        "What muscles does deadlift work?",
        "I need motivation to continue",
        "My knee hurts when I squat"  # Medical query - should trigger disclaimer
    ]
    
    user_id = "user_123"
    
    for query in test_queries:
        print(f"\nUser: {query}")
        response = chatbot.process_query(user_id, query)
        print(f"Bot: {response['response']}")
        print(f"Intent: {response['intent']}")
        print(f"Entities: {response['entities']}")