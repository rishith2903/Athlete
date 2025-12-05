# Phase 3 – Backend APIs Testing Report

Instructions
- Ensure DB is running and seeded. For current repo, backend uses MongoDB; if PostgreSQL is desired, confirm migration first.
- Execute: pwsh scripts/qa/phase3_backend.ps1

Results Table
| Endpoint | Input | Expected | Actual | Status | Perf (avg ms) |
|----------|-------|----------|--------|--------|---------------|
| POST /api/auth/register | new user | 201 + JwtResponse |  |  |  |
| POST /api/auth/login | valid creds | 200 + JwtResponse |  |  |  |
| GET /api/workout | with token | 200 + page content |  |  |  |
| POST /api/workout | create workout | 201 + workout |  |  |  |
| POST /api/workout/ai-generate | prefs | 200 + ai plan |  |  |  |
| POST /api/nutrition/ai-plan | prefs | 200 + plan |  |  |  |
| POST /api/progress | entry | 201 |  |  |  |
| Security: invalid token | any | 401 |  |  |  |
| Security: SQLi payload | auth/login | 401 |  |  |  |
| Load (GET /api/workout) | 500-1000 req | <1s avg |  |  |  |
