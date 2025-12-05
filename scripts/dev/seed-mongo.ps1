Param(
  [string]$BaseUrl = "http://localhost:8080"
)

# Demo user
$registerBody = @{
  username = "demo_user"
  email    = "demo_user@example.com"
  password = "DemoPass123!"
  firstName = "Demo"
  lastName  = "User"
  height = 175.0
  weight = 70.0
  gender = "MALE"
  activityLevel = "MODERATE"
  fitnessGoal = "GENERAL_FITNESS"
} | ConvertTo-Json

Write-Host "Registering demo user..."
try {
  $reg = Invoke-RestMethod -Uri "$BaseUrl/api/auth/register" -Method POST -Body $registerBody -ContentType 'application/json' -ErrorAction Stop
  $token = $reg.accessToken
} catch {
  Write-Host "Registration failed (could be existing), attempting login..."
  $loginBody = @{ usernameOrEmail = "demo_user@example.com"; password = "DemoPass123!" } | ConvertTo-Json
  $login = Invoke-RestMethod -Uri "$BaseUrl/api/auth/login" -Method POST -Body $loginBody -ContentType 'application/json' -ErrorAction Stop
  $token = $login.accessToken
}

if (-not $token) { throw "Cannot obtain JWT token" }
$headers = @{ Authorization = "Bearer $token" }

# Create a sample workout
$workout = @{
  name = "Demo Full Body"
  description = "AI-seeded full body routine"
  type = "MIXED"
  difficulty = "INTERMEDIATE"
  duration = 40
  caloriesBurned = 320
  exercises = @(
    @{ name = "Push-ups"; sets = 3; reps = 12; restTime = 60 },
    @{ name = "Squats"; sets = 4; reps = 15; restTime = 90 },
    @{ name = "Plank"; sets = 3; duration = 45; restTime = 60 }
  )
} | ConvertTo-Json -Depth 5

Write-Host "Creating sample workout..."
Invoke-RestMethod -Uri "$BaseUrl/api/workout" -Method POST -Headers $headers -Body $workout -ContentType 'application/json' | Out-Null

Write-Host "Seeding complete. Demo user: demo_user@example.com / DemoPass123!" -ForegroundColor Green
