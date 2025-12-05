# Phase 1 – AI Models Testing Report

Instructions
- Ensure AI services are running:
  - Workout: http://localhost:8001
  - Nutrition: http://localhost:8002
  - Pose: http://localhost:8003
  - Chatbot: http://localhost:8004
- Execute: pwsh scripts/qa/phase1_models.ps1

Results Table
| Test Case ID | Model | Input Summary | Expected Output | Actual Output | Pass/Fail | Notes |
|--------------|-------|---------------|-----------------|---------------|-----------|-------|
| PSE-IMG-001  | Pose  | squat.jpg (good form) | formScore ≥ 0.80, minimal corrections |  |  |  |
| PSE-VID-002  | Pose  | squat.mp4 (bad knees) | corrections.knees present |  |  |  |
| WKT-PROF-001 | Workout | Beginner, weight loss, no equipment | Safe low-impact plan |  |  |  |
| WKT-PROF-002 | Workout | Advanced, muscle gain, dumbbells | Hypertrophy-focused plan |  |  |  |
| NUT-ALL-001  | Nutrition | Vegetarian, allergy=peanuts | No peanuts, balanced macros |  |  |  |
| NUT-PREF-002 | Nutrition | High protein | Protein target ≥ 30% |  |  |  |
| CBT-SFT-001  | Chatbot | “How much protein?” | Safe, non-medical claims, guidance |  |  |  |
| CBT-CTX-002  | Chatbot | Follow-up after goal set | Context-aware response |  |  |  |

Observations
- 

Issues
- 
