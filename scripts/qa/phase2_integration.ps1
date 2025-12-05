Param(
  [string]$BaseUrl = "http://localhost:8080",
  [string]$ReportPath = "tests/reports/phase2_integration_report.md"
)

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

# Pose check (valid)
$pose = Invoke-TimedRequest -Method 'POST' -Url "$BaseUrl/api/pose/check" -ContentType 'multipart/form-data' -Body @{ file = Get-Item -Path 'backend/models/api_services/sample_pose.jpg' -ErrorAction SilentlyContinue; exerciseType = 'squat' }
Write-Host "POSE valid:" ($pose | ConvertTo-Json -Depth 5)

# Pose check (invalid)
$poseBad = Invoke-TimedRequest -Method 'POST' -Url "$BaseUrl/api/pose/check" -ContentType 'multipart/form-data' -Body @{ file = $null; exerciseType = 'squat' }
Write-Host "POSE invalid:" ($poseBad | ConvertTo-Json -Depth 5)

# Workout AI generate
$wkt = Invoke-TimedRequest -Method 'POST' -Url "$BaseUrl/api/workout/ai-generate" -Body @{ fitnessGoal='weight_loss'; activityLevel='BEGINNER'; equipment=@(); workoutDuration=30 }
Write-Host "WORKOUT gen:" ($wkt | ConvertTo-Json -Depth 5)

# Nutrition AI plan
$nut = Invoke-TimedRequest -Method 'POST' -Url "$BaseUrl/api/nutrition/ai-plan" -Body @{ preferences=@{ dietType='vegetarian' } }
Write-Host "NUTRITION plan:" ($nut | ConvertTo-Json -Depth 5)

# Chatbot
$cbt = Invoke-TimedRequest -Method 'POST' -Url "$BaseUrl/api/chatbot" -Body @{ message='Tips for building muscle?' }
Write-Host "CHATBOT:" ($cbt | ConvertTo-Json -Depth 5)
