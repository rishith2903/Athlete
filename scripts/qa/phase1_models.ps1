Param(
  [string]$ReportPath = "tests/reports/phase1_models_report.md"
)

# Utilities
function Write-Section($title) {
  Write-Host "`n=== $title ===" -ForegroundColor Cyan
}

function Invoke-TimedRequest {
  Param(
    [string]$Method,
    [string]$Url,
    [hashtable]$Headers = @{},
    [object]$Body = $null,
    [string]$ContentType = 'application/json'
  )
  $start = Get-Date
  try {
    $params = @{ Uri = $Url; Method = $Method; Headers = $Headers; ErrorAction = 'Stop' }
    if ($Body -ne $null) {
      if ($ContentType -eq 'multipart/form-data') {
        $params['Form'] = $Body
      } else {
        $params['ContentType'] = $ContentType
        $params['Body'] = ($Body | ConvertTo-Json -Depth 6)
      }
    }
    $resp = Invoke-RestMethod @params
    $latency = [math]::Round(((Get-Date) - $start).TotalMilliseconds, 2)
    return @{ ok=$true; data=$resp; latency=$latency }
  } catch {
    $latency = [math]::Round(((Get-Date) - $start).TotalMilliseconds, 2)
    return @{ ok=$false; error=$_.Exception.Message; latency=$latency; raw=$_ }
  }
}

# Pose: image
Write-Section "Pose – Image"
$poseImg = Invoke-TimedRequest -Method 'POST' -Url 'http://localhost:8003/analyze' -ContentType 'multipart/form-data' -Body @{ file = Get-Item -Path 'backend/models/api_services/sample_pose.jpg' -ErrorAction SilentlyContinue; exercise_type = 'squat' }
Write-Host ($poseImg | ConvertTo-Json -Depth 5)

# Workout: beginner profile
Write-Section "Workout – Beginner"
$wktBeg = Invoke-TimedRequest -Method 'POST' -Url 'http://localhost:8001/recommend' -Body @{ userId='qa'; fitnessGoal='weight_loss'; activityLevel='beginner'; equipment=@(); workoutDuration=30 }
Write-Host ($wktBeg | ConvertTo-Json -Depth 5)

# Nutrition: vegetarian
Write-Section "Nutrition – Vegetarian"
$nutVeg = Invoke-TimedRequest -Method 'POST' -Url 'http://localhost:8002/plan' -Body @{ userId='qa'; preferences=@{ dietType='vegetarian' } }
Write-Host ($nutVeg | ConvertTo-Json -Depth 5)

# Chatbot: safe prompt
Write-Section "Chatbot – Safety"
$cbt = Invoke-TimedRequest -Method 'POST' -Url 'http://localhost:8004/api/chat/message' -Body @{ user_id='qa'; message='How much protein should I eat daily?' }
Write-Host ($cbt | ConvertTo-Json -Depth 5)

Write-Host "`nFill in results in $ReportPath based on the outputs above." -ForegroundColor Yellow
