# Phase 2 – Backend ↔ AI Integration Testing Report

Instructions
- Ensure Spring backend (http://localhost:8080) and AI services (8001–8004) are running.
- Execute: pwsh scripts/qa/phase2_integration.ps1

Results Table
| Endpoint | Input | Expected Response | Actual Response | Pass/Fail | Latency (ms) |
|----------|-------|-------------------|-----------------|-----------|--------------|
| POST /api/pose/check | file=squat.jpg, exerciseType=squat | 200, {formScore, feedback, corrections} |  |  |  |
| POST /api/workout/ai-generate | profile=beginner | 200, workout plan JSON |  |  |  |
| POST /api/nutrition/ai-plan | preferences=vegetarian | 200, meal plan JSON |  |  |  |
| POST /api/chatbot | message="How to gain muscle?" | 200, response, intent |  |  |  |
| TIMEOUT /api/chatbot | simulate timeout | 503 or 504, error JSON |  |  |  |
| INVALID /api/pose/check | empty file | 400, error JSON |  |  |  |

Observations
- 

Issues
- 
