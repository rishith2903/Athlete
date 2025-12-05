# Phase 6 – Unit Testing Report

Instructions
- Backend: mvn -f backend/pom.xml test (with coverage if configured)
- Frontend: npm --prefix frontend run test (Vitest) and npm --prefix frontend run test:e2e (Playwright)
- AI services: pytest

Results Table
| Module/Function | Test Name | Input | Expected → Actual | Pass/Fail | Coverage % |
|-----------------|-----------|-------|-------------------|-----------|------------|
| AuthService     | register_validUser_createsAndReturnsJwt | new user payload | 201 + tokens →  |  |  |
| AuthService     | login_invalid_creds_returns401 | wrong creds | 401 →  |  |  |
| AuthContext     | persists_token_and_user | token + profile | localStorage updated →  |  |  |
| api.js          | sets_auth_header | token | Authorization header present →  |  |  |
| pose_service    | analyze_returns_schema | image | {formScore, feedback, corrections} →  |  |  |

Coverage Summary
- Backend:   
- Frontend:  
- AI:        
