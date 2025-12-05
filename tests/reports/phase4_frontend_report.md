# Phase 4 – Frontend (React) Testing Report

Instructions
- Run dev server at http://localhost:5173 and backend at http://localhost:8080.
- Execute: npm --prefix frontend run test:e2e (for E2E) and manual cross-browser runs.

Results Table
| Component/Page | Browser/Device | Test Case | Expected | Actual | Pass/Fail |
|----------------|----------------|-----------|----------|--------|-----------|
| Login/Register | Chrome Desktop | Valid login | Redirect to /dashboard |  |  |
| Dashboard      | Chrome Mobile  | Charts render | No console errors, responsive |  |  |
| Pose Analysis  | Edge Desktop   | Camera perms | Video displays, analysis snapshots |  |  |
| Chatbot        | Firefox        | Send/receive | Response rendered with timestamp |  |  |
| Nutrition      | Safari (macOS) | Generate plan | Macros and meals visible |  |  |
| A11y           | Chromium + Axe | WCAG checks | No violations (or minor) |  |  |

Issues
- 
