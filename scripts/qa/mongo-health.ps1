Param(
  [string]$MongoUri = "mongodb://localhost:27017/fitness_db"
)

Write-Host "Checking MongoDB connectivity..." -ForegroundColor Cyan
try {
  $client = New-Object MongoDB.Driver.MongoClient($MongoUri)
  $db = $client.GetDatabase('fitness_db')
  $collectionNames = $db.ListCollectionNames().ToList()
  Write-Host "Connected. Collections: $($collectionNames -join ', ')"
  Write-Host "OK" -ForegroundColor Green
} catch {
  Write-Host "Failed to connect: $($_.Exception.Message)" -ForegroundColor Red
  exit 1
}
