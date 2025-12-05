Param(
  [string]$BaseUrl = "http://localhost:8080",
  [int]$Concurrent = 500,
  [int]$RequestsPerUser = 2
)

$results = [System.Collections.Concurrent.ConcurrentBag[double]]::new()
$errors = [System.Collections.Concurrent.ConcurrentBag[string]]::new()

$scriptBlock = {
  Param($BaseUrl, $RequestsPerUser)
  for ($i=0; $i -lt $RequestsPerUser; $i++) {
    $start = Get-Date
    try {
      $resp = Invoke-RestMethod -Uri "$BaseUrl/api/workout" -Method GET -ErrorAction Stop
      $lat = ((Get-Date) - $start).TotalMilliseconds
      return ,@($lat)
    } catch {
      return ,@(-1)
    }
  }
}

$jobs = @()
for ($j=0; $j -lt $Concurrent; $j++) {
  $jobs += Start-Job -ScriptBlock $scriptBlock -ArgumentList $BaseUrl, $RequestsPerUser
}

Wait-Job -Job $jobs | Out-Null
foreach ($job in $jobs) {
  $vals = Receive-Job -Job $job
  foreach ($v in $vals) {
    if ($v -ge 0) { $results.Add([double]$v) } else { $errors.Add("error") }
  }
}

if ($results.Count -gt 0) {
  $avg = ($results | Measure-Object -Average).Average
  $min = ($results | Measure-Object -Minimum).Minimum
  $max = ($results | Measure-Object -Maximum).Maximum
  Write-Host ("Requests: {0}, Success: {1}, Errors: {2}" -f ($Concurrent*$RequestsPerUser), $results.Count, $errors.Count)
  Write-Host ("Latency ms -> avg: {0:N2}, min: {1:N2}, max: {2:N2}" -f $avg, $min, $max)
} else {
  Write-Host "No successful requests. Errors: $($errors.Count)" -ForegroundColor Red
}
