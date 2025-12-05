# Backend QA Test Script
# Tests the Spring Boot backend REST APIs

$baseUrl = "http://localhost:8080"
$script:testResults = @()
$testEmail = "qatest_$(Get-Random)@example.com"
$testPassword = "Test123!@#"
$authToken = ""

function Test-Endpoint {
    param(
        [string]$Name,
        [string]$Method,
        [string]$Endpoint,
        [hashtable]$Headers = @{},
        [string]$Body = "",
        [int]$ExpectedStatus = 200
    )
    
    Write-Host "`n--- TEST: $Name ---" -ForegroundColor Cyan
    Write-Host "Endpoint: $Method $Endpoint"
    
    if ($Body) {
        Write-Host "Request Body: $Body"
    }
    
    $startTime = Get-Date
    
    try {
        $params = @{
            Uri = "$baseUrl$Endpoint"
            Method = $Method
            Headers = $Headers
            ContentType = "application/json"
        }
        
        if ($Body) {
            $params.Body = $Body
        }
        
        $response = Invoke-RestMethod @params -ErrorAction Stop
        # Assume success if we get here
        $statusCode = 200
        
        $endTime = Get-Date
        $responseTime = [math]::Round(($endTime - $startTime).TotalMilliseconds, 2)
        
        Write-Host "Expected Status: $ExpectedStatus"
        Write-Host "Actual Status: $statusCode"
        Write-Host "Response Time: $responseTime ms"
        
        if ($statusCode -eq $ExpectedStatus) {
            Write-Host "Status: PASS" -ForegroundColor Green
            $result = "PASS"
        } else {
            Write-Host "Status: FAIL" -ForegroundColor Red
            $result = "FAIL"
        }
        
        $script:testResults += @{
            Test = $Name
            Endpoint = "$Method $Endpoint"
            ExpectedStatus = $ExpectedStatus
            ActualStatus = $statusCode
            ResponseTime = $responseTime
            Result = $result
            Response = $response
        }
        
        return $response
        
    } catch {
        $endTime = Get-Date
        $responseTime = [math]::Round(($endTime - $startTime).TotalMilliseconds, 2)
        
        Write-Host "Error: $_" -ForegroundColor Red
        Write-Host "Status: FAIL" -ForegroundColor Red
        
        $script:testResults += @{
            Test = $Name
            Endpoint = "$Method $Endpoint"
            ExpectedStatus = $ExpectedStatus
            ActualStatus = "ERROR"
            ResponseTime = $responseTime
            Result = "FAIL"
            Error = $_.Exception.Message
        }
        
        return $null
    }
}

Write-Host "================================================================================`n" -ForegroundColor Yellow
Write-Host "BACKEND QA TEST SUITE - COMPREHENSIVE TESTING" -ForegroundColor Yellow
Write-Host "Test Started: $(Get-Date)" -ForegroundColor Yellow
Write-Host "`n================================================================================" -ForegroundColor Yellow

# Check if backend is running
Write-Host "`nChecking backend availability..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "$baseUrl/actuator/health" -ErrorAction SilentlyContinue
    if ($health.status -eq "UP") {
        Write-Host "Backend is UP and running!" -ForegroundColor Green
    }
} catch {
    Write-Host "Warning: Backend health check failed. Trying basic endpoints..." -ForegroundColor Yellow
}

# ==================== AUTHENTICATION TESTS ====================
Write-Host "`n==================== AUTHENTICATION TESTS ====================" -ForegroundColor Magenta

# Test 1: User Registration
$registerBody = @{
    username = "qatestuser_$(Get-Random)"
    email = $testEmail
    password = $testPassword
    firstName = "QA"
    lastName = "Tester"
    height = 175.0
    weight = 70.0
    gender = "MALE"
    activityLevel = "MODERATE"
    fitnessGoal = "LOSE_WEIGHT"
} | ConvertTo-Json

$registerResponse = Test-Endpoint -Name "User Registration (Valid)" `
    -Method "POST" `
    -Endpoint "/api/auth/register" `
    -Body $registerBody `
    -ExpectedStatus 201

if ($registerResponse.accessToken) {
    $authToken = $registerResponse.accessToken
    $tokenPreview = if ($authToken.Length -gt 20) { $authToken.Substring(0, 20) + "..." } else { $authToken }
    Write-Host "Auth token obtained: $tokenPreview" -ForegroundColor Green
}

# Test 2: User Login
$loginBody = @{
    usernameOrEmail = $testEmail
    password = $testPassword
} | ConvertTo-Json

$loginResponse = Test-Endpoint -Name "User Login (Valid)" `
    -Method "POST" `
    -Endpoint "/api/auth/login" `
    -Body $loginBody `
    -ExpectedStatus 200

if ($loginResponse.accessToken -and -not $authToken) {
    $authToken = $loginResponse.accessToken
}

# Test 3: Invalid Login
$invalidLoginBody = @{
    usernameOrEmail = $testEmail
    password = "WrongPassword123!"
} | ConvertTo-Json

Test-Endpoint -Name "User Login (Invalid)" `
    -Method "POST" `
    -Endpoint "/api/auth/login" `
    -Body $invalidLoginBody `
    -ExpectedStatus 401

# ==================== WORKOUT MANAGEMENT TESTS ====================
Write-Host "`n==================== WORKOUT MANAGEMENT TESTS ====================" -ForegroundColor Magenta

$authHeaders = @{
    "Authorization" = "Bearer $authToken"
}

# Test 4: Create Workout
$workoutBody = @{
    name = "Morning Cardio"
    type = "CARDIO"
    difficulty = "INTERMEDIATE"
    duration = 30
    exercises = @(
        @{
            name = "Running"
            duration = 20
            sets = 1
        },
        @{
            name = "Jumping Jacks"
            duration = 10
            sets = 3
        }
    )
} | ConvertTo-Json -Depth 3

Test-Endpoint -Name "Create Workout" `
    -Method "POST" `
    -Endpoint "/api/workout" `
    -Headers $authHeaders `
    -Body $workoutBody `
    -ExpectedStatus 201

# Test 5: Get Workouts
Test-Endpoint -Name "Get User Workouts" `
    -Method "GET" `
    -Endpoint "/api/workout" `
    -Headers $authHeaders `
    -ExpectedStatus 200

# ==================== NUTRITION TESTS ====================
Write-Host "`n==================== NUTRITION TESTS ====================" -ForegroundColor Magenta

# Test 6: Log Meal
$mealBody = @{
    name = "Healthy Breakfast"
    type = "BREAKFAST"
    date = (Get-Date).ToString("yyyy-MM-dd")
    totalCalories = 450
    totalProtein = 25
    totalCarbs = 55
    totalFat = 15
    foodItems = @(
        @{
            name = "Oatmeal"
            calories = 300
            quantity = 100
            unit = "g"
        },
        @{
            name = "Banana"
            calories = 150
            quantity = 1
            unit = "piece"
        }
    )
} | ConvertTo-Json -Depth 3

Test-Endpoint -Name "Log Meal" `
    -Method "POST" `
    -Endpoint "/api/nutrition" `
    -Headers $authHeaders `
    -Body $mealBody `
    -ExpectedStatus 201

# ==================== PROGRESS TRACKING TESTS ====================
Write-Host "`n==================== PROGRESS TRACKING TESTS ====================" -ForegroundColor Magenta

# Test 7: Log Progress
$progressBody = @{
    date = (Get-Date).ToString("yyyy-MM-dd")
    weight = 69.5
    bodyFatPercentage = 18.5
    workoutsCompleted = 5
    caloriesConsumed = 2100
    mood = "GOOD"
    energyLevel = 8
} | ConvertTo-Json

Test-Endpoint -Name "Log Progress" `
    -Method "POST" `
    -Endpoint "/api/progress" `
    -Headers $authHeaders `
    -Body $progressBody `
    -ExpectedStatus 201

# ==================== SECURITY TESTS ====================
Write-Host "`n==================== SECURITY TESTS ====================" -ForegroundColor Magenta

# Test 8: Access without token
Test-Endpoint -Name "Unauthorized Access (No Token)" `
    -Method "GET" `
    -Endpoint "/api/workout" `
    -ExpectedStatus 401

# Test 9: Invalid token
$invalidHeaders = @{
    "Authorization" = "Bearer invalid.token.here"
}

Test-Endpoint -Name "Unauthorized Access (Invalid Token)" `
    -Method "GET" `
    -Endpoint "/api/workout" `
    -Headers $invalidHeaders `
    -ExpectedStatus 401

# Test 10: SQL Injection attempt
$sqlInjectionBody = @{
    usernameOrEmail = "'; DROP TABLE users; --"
    password = "password"
} | ConvertTo-Json

Test-Endpoint -Name "SQL Injection Prevention" `
    -Method "POST" `
    -Endpoint "/api/auth/login" `
    -Body $sqlInjectionBody `
    -ExpectedStatus 401

# ==================== ERROR HANDLING TESTS ====================
Write-Host "`n==================== ERROR HANDLING TESTS ====================" -ForegroundColor Magenta

# Test 11: 404 Not Found
Test-Endpoint -Name "404 Not Found" `
    -Method "GET" `
    -Endpoint "/api/workout/nonexistent-id" `
    -Headers $authHeaders `
    -ExpectedStatus 404

# Test 12: Bad Request
$invalidJson = "{ invalid json }"
Test-Endpoint -Name "400 Bad Request (Invalid JSON)" `
    -Method "POST" `
    -Endpoint "/api/workout" `
    -Headers $authHeaders `
    -Body $invalidJson `
    -ExpectedStatus 400

# ==================== PAGINATION TESTS ====================
Write-Host "`n==================== PAGINATION TESTS ====================" -ForegroundColor Magenta

# Test 13: Pagination
Test-Endpoint -Name "Pagination (page=0, size=5)" `
    -Method "GET" `
    -Endpoint "/api/workout?page=0&size=5" `
    -Headers $authHeaders `
    -ExpectedStatus 200

# ==================== LOAD TESTS ====================
Write-Host "`n==================== LOAD/STRESS TESTS ====================" -ForegroundColor Magenta

Write-Host "`nRunning concurrent load test..." -ForegroundColor Yellow
$loadTestResults = @()
$numberOfRequests = 50

Write-Host "Simulating $numberOfRequests rapid requests..."

for ($i = 1; $i -le $numberOfRequests; $i++) {
    $startTime = Get-Date
    try {
        $response = Invoke-RestMethod -Uri "$baseUrl/api/workout" `
            -Headers $authHeaders `
            -Method GET `
            -ErrorAction SilentlyContinue
        
        $endTime = Get-Date
        $responseTime = [math]::Round(($endTime - $startTime).TotalMilliseconds, 2)
        $loadTestResults += $responseTime
        
        if ($i % 10 -eq 0) {
            Write-Host "Completed $i requests..." -NoNewline
            Write-Host " (Last response: $responseTime ms)" -ForegroundColor Gray
        }
    } catch {
        # Ignore errors during load test
    }
}

if ($loadTestResults.Count -gt 0) {
    $avgResponseTime = [math]::Round(($loadTestResults | Measure-Object -Average).Average, 2)
    $minResponseTime = ($loadTestResults | Measure-Object -Minimum).Minimum
    $maxResponseTime = ($loadTestResults | Measure-Object -Maximum).Maximum
    
    Write-Host "`nLoad Test Results:" -ForegroundColor Cyan
    Write-Host "Total Requests: $numberOfRequests"
    Write-Host "Successful Requests: $($loadTestResults.Count)"
    Write-Host "Average Response Time: $avgResponseTime ms"
    Write-Host "Min Response Time: $minResponseTime ms"
    Write-Host "Max Response Time: $maxResponseTime ms"
    
    if ($avgResponseTime -lt 1000) {
        Write-Host "Performance: PASS (avg < 1s)" -ForegroundColor Green
    } else {
        Write-Host "Performance: NEEDS IMPROVEMENT (avg > 1s)" -ForegroundColor Yellow
    }
}

# ==================== TEST SUMMARY ====================
Write-Host "`n================================================================================`n" -ForegroundColor Yellow
Write-Host "TEST SUMMARY" -ForegroundColor Yellow
Write-Host "================================================================================`n" -ForegroundColor Yellow

$passCount = ($testResults | Where-Object { $_.Result -eq "PASS" }).Count
$failCount = ($testResults | Where-Object { $_.Result -eq "FAIL" }).Count
$totalTests = $testResults.Count

Write-Host "Total Tests Run: $totalTests"
Write-Host "Passed: $passCount" -ForegroundColor Green
Write-Host "Failed: $failCount" -ForegroundColor Red
if ($totalTests -gt 0) {
    Write-Host "Success Rate: $([math]::Round(($passCount / $totalTests) * 100, 2))%"
} else {
    Write-Host "Success Rate: N/A (no tests completed)"
}

Write-Host "`nDetailed Results:" -ForegroundColor Cyan
$testResults | ForEach-Object {
    $color = if ($_.Result -eq "PASS") { "Green" } else { "Red" }
    Write-Host "$($_.Test): $($_.Result)" -ForegroundColor $color
    Write-Host "  Endpoint: $($_.Endpoint)"
    Write-Host "  Response Time: $($_.ResponseTime) ms"
    if ($_.Error) {
        Write-Host "  Error: $($_.Error)" -ForegroundColor Red
    }
}

Write-Host "`n================================================================================`n" -ForegroundColor Yellow
Write-Host "QA TEST SUITE COMPLETED" -ForegroundColor Yellow
Write-Host "Test Ended: $(Get-Date)" -ForegroundColor Yellow
Write-Host "`n================================================================================`n" -ForegroundColor Yellow