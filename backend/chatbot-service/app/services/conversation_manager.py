"""
Conversation Manager - Handles session management and context preservation
"""

import redis.asyncio as redis
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import os
from collections import deque

logger = logging.getLogger(__name__)

class ConversationManager:
    def __init__(self):
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", 6379))
        self.redis_client = None
        self.conversation_ttl = 3600 * 24  # 24 hours
        self.max_history_length = 20  # Keep last 20 messages
        
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = await redis.from_url(
                f"redis://{self.redis_host}:{self.redis_port}",
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis connection established for conversation management")
        except Exception as e:
            logger.warning(f"Redis connection failed: {str(e)}. Using in-memory storage.")
            self.redis_client = None
            self.memory_storage = {}
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.redis_client:
            await self.redis_client.close()
    
    async def get_context(self, user_id: str) -> Dict:
        """Get user conversation context"""
        try:
            if self.redis_client:
                # Get from Redis
                context_key = f"context:{user_id}"
                context_data = await self.redis_client.get(context_key)
                
                if context_data:
                    return json.loads(context_data)
            else:
                # Get from memory
                return self.memory_storage.get(f"context:{user_id}", {})
            
            # Return default context if none exists
            return await self.create_default_context(user_id)
            
        except Exception as e:
            logger.error(f"Error getting context for user {user_id}: {str(e)}")
            return {}
    
    async def update_context(self, user_id: str, context: Dict):
        """Update user conversation context"""
        try:
            context_key = f"context:{user_id}"
            context_data = json.dumps(context)
            
            if self.redis_client:
                await self.redis_client.setex(
                    context_key,
                    self.conversation_ttl,
                    context_data
                )
            else:
                self.memory_storage[context_key] = context
                
        except Exception as e:
            logger.error(f"Error updating context for user {user_id}: {str(e)}")
    
    async def create_default_context(self, user_id: str) -> Dict:
        """Create default context for new user"""
        context = {
            "user_id": user_id,
            "session_start": datetime.utcnow().isoformat(),
            "message_count": 0,
            "last_intent": None,
            "user_profile": {
                "fitness_level": None,
                "goals": [],
                "preferences": {},
                "injuries": []
            },
            "conversation_state": "active",
            "last_workout": None,
            "pending_actions": []
        }
        
        await self.update_context(user_id, context)
        return context
    
    async def add_message(self, user_id: str, user_message: str, bot_response: Dict):
        """Add message to conversation history"""
        try:
            # Get current history
            history = await self.get_conversation_history(user_id)
            
            # Add new message pair
            message_pair = {
                "timestamp": datetime.utcnow().isoformat(),
                "user_message": user_message,
                "bot_response": bot_response["response"],
                "intent": bot_response.get("intent"),
                "entities": bot_response.get("entities", [])
            }
            
            history.append(message_pair)
            
            # Trim history if too long
            if len(history) > self.max_history_length:
                history = history[-self.max_history_length:]
            
            # Save history
            history_key = f"history:{user_id}"
            
            if self.redis_client:
                await self.redis_client.setex(
                    history_key,
                    self.conversation_ttl,
                    json.dumps(history)
                )
            else:
                self.memory_storage[history_key] = history
            
            # Update context
            context = await self.get_context(user_id)
            context["message_count"] = context.get("message_count", 0) + 1
            context["last_intent"] = bot_response.get("intent")
            context["last_message_time"] = datetime.utcnow().isoformat()
            
            # Extract and update user profile from conversation
            await self.extract_user_info(user_id, user_message, bot_response.get("entities", []))
            
            await self.update_context(user_id, context)
            
        except Exception as e:
            logger.error(f"Error adding message for user {user_id}: {str(e)}")
    
    async def get_conversation_history(self, user_id: str) -> List[Dict]:
        """Get conversation history for user"""
        try:
            history_key = f"history:{user_id}"
            
            if self.redis_client:
                history_data = await self.redis_client.get(history_key)
                if history_data:
                    return json.loads(history_data)
            else:
                return self.memory_storage.get(history_key, [])
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting history for user {user_id}: {str(e)}")
            return []
    
    async def extract_user_info(self, user_id: str, message: str, entities: List[Dict]):
        """Extract and update user information from conversation"""
        try:
            context = await self.get_context(user_id)
            user_profile = context.get("user_profile", {})
            
            # Extract fitness level
            if "beginner" in message.lower():
                user_profile["fitness_level"] = "beginner"
            elif "intermediate" in message.lower():
                user_profile["fitness_level"] = "intermediate"
            elif "advanced" in message.lower():
                user_profile["fitness_level"] = "advanced"
            
            # Extract goals
            goals_keywords = {
                "weight loss": "weight_loss",
                "lose weight": "weight_loss",
                "muscle gain": "muscle_gain",
                "build muscle": "muscle_gain",
                "strength": "strength",
                "endurance": "endurance",
                "flexibility": "flexibility",
                "tone": "toning"
            }
            
            for keyword, goal in goals_keywords.items():
                if keyword in message.lower():
                    if goal not in user_profile.get("goals", []):
                        user_profile.setdefault("goals", []).append(goal)
            
            # Extract preferences from entities
            for entity in entities:
                if entity["type"] == "exercise":
                    user_profile.setdefault("preferences", {}).setdefault("exercises", [])
                    if entity["text"] not in user_profile["preferences"]["exercises"]:
                        user_profile["preferences"]["exercises"].append(entity["text"])
                
                elif entity["type"] == "food":
                    user_profile.setdefault("preferences", {}).setdefault("foods", [])
                    if entity["text"] not in user_profile["preferences"]["foods"]:
                        user_profile["preferences"]["foods"].append(entity["text"])
            
            # Check for injuries or limitations
            injury_keywords = ["injury", "injured", "pain", "hurt", "avoid"]
            if any(keyword in message.lower() for keyword in injury_keywords):
                # Mark that user might have limitations
                user_profile["has_limitations"] = True
            
            context["user_profile"] = user_profile
            await self.update_context(user_id, context)
            
        except Exception as e:
            logger.error(f"Error extracting user info: {str(e)}")
    
    async def get_conversation_summary(self, user_id: str) -> str:
        """Generate a summary of the conversation"""
        try:
            history = await self.get_conversation_history(user_id)
            context = await self.get_context(user_id)
            
            if not history:
                return "No conversation history available."
            
            # Create summary
            summary = f"Conversation Summary for User {user_id}:\n"
            summary += f"Total messages: {len(history)}\n"
            
            # Count intents
            intent_counts = {}
            for msg in history:
                intent = msg.get("intent")
                if intent:
                    intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
            summary += "Topics discussed:\n"
            for intent, count in intent_counts.items():
                summary += f"  - {intent.replace('_', ' ').title()}: {count} times\n"
            
            # Add user profile info
            user_profile = context.get("user_profile", {})
            if user_profile.get("fitness_level"):
                summary += f"Fitness Level: {user_profile['fitness_level']}\n"
            if user_profile.get("goals"):
                summary += f"Goals: {', '.join(user_profile['goals'])}\n"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Error generating conversation summary."
    
    async def clear_conversation(self, user_id: str):
        """Clear conversation history and context for user"""
        try:
            if self.redis_client:
                await self.redis_client.delete(f"context:{user_id}")
                await self.redis_client.delete(f"history:{user_id}")
            else:
                self.memory_storage.pop(f"context:{user_id}", None)
                self.memory_storage.pop(f"history:{user_id}", None)
            
            logger.info(f"Cleared conversation for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error clearing conversation: {str(e)}")
    
    async def is_session_active(self, user_id: str) -> bool:
        """Check if user session is still active"""
        try:
            context = await self.get_context(user_id)
            if not context:
                return False
            
            last_message_time = context.get("last_message_time")
            if not last_message_time:
                return False
            
            # Check if last message was within 30 minutes
            last_time = datetime.fromisoformat(last_message_time)
            time_diff = datetime.utcnow() - last_time
            
            return time_diff < timedelta(minutes=30)
            
        except Exception as e:
            logger.error(f"Error checking session status: {str(e)}")
            return False
    
    async def get_user_stats(self, user_id: str) -> Dict:
        """Get user conversation statistics"""
        try:
            history = await self.get_conversation_history(user_id)
            context = await self.get_context(user_id)
            
            if not history:
                return {
                    "total_messages": 0,
                    "session_count": 0,
                    "avg_session_length": 0
                }
            
            # Calculate stats
            total_messages = len(history)
            
            # Group messages by session (30 min gap = new session)
            sessions = []
            current_session = []
            last_time = None
            
            for msg in history:
                msg_time = datetime.fromisoformat(msg["timestamp"])
                
                if last_time and (msg_time - last_time) > timedelta(minutes=30):
                    if current_session:
                        sessions.append(current_session)
                    current_session = [msg]
                else:
                    current_session.append(msg)
                
                last_time = msg_time
            
            if current_session:
                sessions.append(current_session)
            
            avg_session_length = sum(len(s) for s in sessions) / len(sessions) if sessions else 0
            
            return {
                "total_messages": total_messages,
                "session_count": len(sessions),
                "avg_session_length": round(avg_session_length, 1),
                "user_profile": context.get("user_profile", {}),
                "most_discussed_topics": self.get_top_topics(history)
            }
            
        except Exception as e:
            logger.error(f"Error getting user stats: {str(e)}")
            return {}
    
    def get_top_topics(self, history: List[Dict], top_n: int = 3) -> List[str]:
        """Get most discussed topics from history"""
        intent_counts = {}
        
        for msg in history:
            intent = msg.get("intent")
            if intent:
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        sorted_intents = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [intent.replace('_', ' ').title() for intent, _ in sorted_intents[:top_n]]