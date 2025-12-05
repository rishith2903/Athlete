"""
API Routes for Chatbot Service
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import logging
from datetime import datetime

from app.models.chatbot_model import ChatbotModel
from app.services.conversation_manager import ConversationManager
from app.services.backend_integration import BackendIntegrationService

logger = logging.getLogger(__name__)

# Request/Response Models
class ChatMessage(BaseModel):
    user_id: str = Field(..., description="User ID")
    message: str = Field(..., description="User message")
    token: Optional[str] = Field(None, description="Auth token for backend integration")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Bot response")
    intent: Optional[str] = Field(None, description="Detected intent")
    intent_confidence: Optional[float] = Field(None, description="Intent confidence score")
    entities: Optional[List[Dict]] = Field(default_factory=list, description="Extracted entities")
    suggestions: Optional[List[str]] = Field(default_factory=list, description="Follow-up suggestions")
    timestamp: str = Field(..., description="Response timestamp")

class ConversationHistory(BaseModel):
    user_id: str
    messages: List[Dict]
    summary: str

class UserStats(BaseModel):
    user_id: str
    total_messages: int
    session_count: int
    avg_session_length: float
    user_profile: Dict
    most_discussed_topics: List[str]

# Create routers
chat_router = APIRouter()
health_router = APIRouter()

# Dependency injection
async def get_chatbot_model():
    from app.main import chatbot_model
    if not chatbot_model:
        raise HTTPException(status_code=503, detail="Chatbot model not initialized")
    return chatbot_model

async def get_conversation_manager():
    from app.main import conversation_manager
    if not conversation_manager:
        raise HTTPException(status_code=503, detail="Conversation manager not initialized")
    return conversation_manager

async def get_backend_service():
    from app.main import backend_service
    if not backend_service:
        raise HTTPException(status_code=503, detail="Backend service not initialized")
    return backend_service

# Chat endpoints
@chat_router.post("/message", response_model=ChatResponse)
async def send_message(
    request: ChatMessage,
    chatbot: ChatbotModel = Depends(get_chatbot_model),
    conv_manager: ConversationManager = Depends(get_conversation_manager)
):
    """Process a chat message and return response"""
    try:
        # Get user context
        context = await conv_manager.get_context(request.user_id)
        
        # Process message
        response = await chatbot.process_message(
            user_id=request.user_id,
            message=request.message,
            context=context
        )
        
        # Save to conversation history
        await conv_manager.add_message(request.user_id, request.message, response)
        
        return ChatResponse(**response)
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing message")

@chat_router.get("/history/{user_id}", response_model=ConversationHistory)
async def get_conversation_history(
    user_id: str,
    conv_manager: ConversationManager = Depends(get_conversation_manager)
):
    """Get conversation history for a user"""
    try:
        history = await conv_manager.get_conversation_history(user_id)
        summary = await conv_manager.get_conversation_summary(user_id)
        
        return ConversationHistory(
            user_id=user_id,
            messages=history,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching conversation history")

@chat_router.delete("/history/{user_id}")
async def clear_conversation(
    user_id: str,
    conv_manager: ConversationManager = Depends(get_conversation_manager)
):
    """Clear conversation history for a user"""
    try:
        await conv_manager.clear_conversation(user_id)
        return {"message": "Conversation cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Error clearing conversation")

@chat_router.get("/stats/{user_id}", response_model=UserStats)
async def get_user_stats(
    user_id: str,
    conv_manager: ConversationManager = Depends(get_conversation_manager)
):
    """Get user conversation statistics"""
    try:
        stats = await conv_manager.get_user_stats(user_id)
        return UserStats(user_id=user_id, **stats)
        
    except Exception as e:
        logger.error(f"Error fetching stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching user statistics")

@chat_router.post("/workout/generate")
async def generate_workout(
    user_id: str,
    token: str,
    preferences: Dict,
    backend: BackendIntegrationService = Depends(get_backend_service)
):
    """Generate AI workout plan"""
    try:
        workout = await backend.create_ai_workout(user_id, token, preferences)
        if workout:
            return workout
        else:
            raise HTTPException(status_code=404, detail="Could not generate workout")
            
    except Exception as e:
        logger.error(f"Error generating workout: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating workout")

@chat_router.post("/nutrition/generate")
async def generate_meal_plan(
    user_id: str,
    token: str,
    preferences: Dict,
    backend: BackendIntegrationService = Depends(get_backend_service)
):
    """Generate AI meal plan"""
    try:
        meal_plan = await backend.create_ai_meal_plan(user_id, token, preferences)
        if meal_plan:
            return meal_plan
        else:
            raise HTTPException(status_code=404, detail="Could not generate meal plan")
            
    except Exception as e:
        logger.error(f"Error generating meal plan: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating meal plan")

@chat_router.get("/progress/{user_id}")
async def get_user_progress(
    user_id: str,
    token: str,
    backend: BackendIntegrationService = Depends(get_backend_service)
):
    """Get user progress data"""
    try:
        progress = await backend.get_user_progress(user_id, token)
        if progress:
            return progress
        else:
            return {"message": "No progress data available"}
            
    except Exception as e:
        logger.error(f"Error fetching progress: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching progress")

@chat_router.get("/session/{user_id}/status")
async def check_session_status(
    user_id: str,
    conv_manager: ConversationManager = Depends(get_conversation_manager)
):
    """Check if user session is active"""
    try:
        is_active = await conv_manager.is_session_active(user_id)
        return {"user_id": user_id, "session_active": is_active}
        
    except Exception as e:
        logger.error(f"Error checking session: {str(e)}")
        raise HTTPException(status_code=500, detail="Error checking session status")

# Health check endpoints
@health_router.get("/")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "Fitness Chatbot",
        "timestamp": datetime.utcnow().isoformat()
    }

@health_router.get("/ready")
async def readiness_check(
    chatbot: ChatbotModel = Depends(get_chatbot_model),
    conv_manager: ConversationManager = Depends(get_conversation_manager)
):
    """Readiness check - verifies all services are ready"""
    try:
        # Check if models are loaded
        if not chatbot.models_loaded:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        return {
            "status": "ready",
            "models_loaded": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service not ready")