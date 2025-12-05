"""
Fitness Chatbot Service - Main Application
NLP-powered conversational AI for fitness guidance
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging
from typing import Dict, List
import os
from dotenv import load_dotenv

from app.models.chatbot_model import ChatbotModel
from app.services.conversation_manager import ConversationManager
from app.services.backend_integration import BackendIntegrationService
from app.api.routes import chat_router, health_router
from app.utils.logging_config import setup_logging

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global instances
chatbot_model = None
conversation_manager = None
backend_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global chatbot_model, conversation_manager, backend_service
    
    logger.info("Starting Fitness Chatbot Service...")
    
    # Initialize services
    chatbot_model = ChatbotModel()
    await chatbot_model.initialize()
    
    conversation_manager = ConversationManager()
    await conversation_manager.initialize()
    
    backend_service = BackendIntegrationService(
        backend_url=os.getenv("BACKEND_URL", "http://localhost:8080")
    )
    
    logger.info("All services initialized successfully")
    
    yield
    
    # Cleanup
    logger.info("Shutting down services...")
    await conversation_manager.cleanup()
    logger.info("Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Fitness Chatbot Service",
    description="NLP-powered conversational AI for fitness guidance",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
app.include_router(health_router, prefix="/health", tags=["health"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Fitness Chatbot",
        "version": "1.0.0",
        "status": "running",
        "features": [
            "Fitness Q&A",
            "Workout guidance",
            "Diet recommendations",
            "Motivational support",
            "Context-aware conversations"
        ]
    }

@app.websocket("/ws/chat/{user_id}")
async def websocket_chat(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    logger.info(f"WebSocket connection established for user: {user_id}")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message = data.get("message", "")
            
            # Process message
            response = await chatbot_model.process_message(
                user_id=user_id,
                message=message,
                context=await conversation_manager.get_context(user_id)
            )
            
            # Update conversation history
            await conversation_manager.add_message(user_id, message, response)
            
            # Send response back to client
            await websocket.send_json({
                "response": response["response"],
                "intent": response.get("intent"),
                "entities": response.get("entities"),
                "suggestions": response.get("suggestions", [])
            })
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user: {user_id}")
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {str(e)}")
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8004)),
        reload=os.getenv("ENV", "development") == "development"
    )