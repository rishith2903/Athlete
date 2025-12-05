# AI Fitness Backend

A robust Spring Boot backend for an AI-powered fitness application that provides user management, workout planning, nutrition tracking, progress monitoring, and AI model integration.

## 🚀 Features

- **JWT Authentication**: Secure user authentication and authorization
- **User Management**: Complete user profile management with fitness goals
- **Workout Planning**: AI-generated personalized workout plans
- **Nutrition Tracking**: AI-powered meal planning and nutrition tracking
- **Form Check**: Exercise form analysis using computer vision
- **Progress Tracking**: Comprehensive progress monitoring and analytics
- **Chatbot Integration**: AI chatbot for fitness advice
- **Real-time Updates**: WebSocket support for real-time features
- **Caching**: Redis caching for improved performance
- **API Documentation**: Swagger/OpenAPI documentation

## 🛠️ Tech Stack

- **Framework**: Spring Boot 3.2.0
- **Language**: Java 17
- **Database**: MongoDB
- **Cache**: Redis
- **Security**: Spring Security + JWT
- **Documentation**: SpringDoc OpenAPI (Swagger)
- **Build Tool**: Maven
- **Deployment**: Docker, Render

## 📋 Prerequisites

- Java 17 or higher
- Maven 3.6+
- MongoDB 6.0+
- Redis 6.0+
- Docker (optional)

## 🔑 Environment

Copy .env.example to .env (optional) and export variables in your shell:

```bash
# PowerShell examples
$env:MONGODB_URI = "mongodb://localhost:27017/fitness_db"
$env:REDIS_HOST = "localhost"
$env:WORKOUT_SERVICE_URL = "http://localhost:8001"
$env:NUTRITION_SERVICE_URL = "http://localhost:8002"
$env:POSE_SERVICE_URL = "http://localhost:8003"
$env:CHATBOT_SERVICE_URL = "http://localhost:8004"
$env:JWT_SECRET = "replace-with-secret"
```

Seed demo data:
```bash
pwsh ../scripts/dev/seed-mongo.ps1
```

## 🔧 Installation

### Local Development

1. **Clone the repository**:
```bash
cd backend
```

2. **Install dependencies**:
```bash
mvn clean install
```

3. **Set up MongoDB**:
```bash
# Using Docker
docker run -d -p 27017:27017 --name mongodb mongo:7.0

# Or install MongoDB locally
```

4. **Set up Redis**:
```bash
# Using Docker
docker run -d -p 6379:6379 --name redis redis:7-alpine

# Or install Redis locally
```

5. **Configure environment variables**:
Create `.env` file in the backend directory:
```env
MONGODB_URI=mongodb://localhost:27017/fitness_db
REDIS_HOST=localhost
REDIS_PORT=6379
JWT_SECRET=your-secret-key-here
```

6. **Run the application**:
```bash
mvn spring-boot:run
```

### Using Docker Compose

1. **Build and run all services**:
```bash
docker-compose up -d
```

2. **View logs**:
```bash
docker-compose logs -f backend
```

3. **Stop services**:
```bash
docker-compose down
```

## 📚 API Documentation

Once the application is running, access the API documentation at:
- Swagger UI: http://localhost:8080/swagger-ui.html
- OpenAPI JSON: http://localhost:8080/api-docs

## 🔑 API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - User login
- `POST /api/auth/refresh` - Refresh access token
- `POST /api/auth/logout` - User logout
- `POST /api/auth/forgot-password` - Request password reset
- `POST /api/auth/reset-password` - Reset password
- `GET /api/auth/verify-email` - Verify email address

### Workout Management
- `GET /api/workout` - Get user workouts
- `POST /api/workout` - Create new workout
- `GET /api/workout/{id}` - Get workout by ID
- `PUT /api/workout/{id}` - Update workout
- `DELETE /api/workout/{id}` - Delete workout
- `POST /api/workout/ai-generate` - Generate AI workout plan

### Nutrition Management
- `GET /api/nutrition` - Get user meals
- `POST /api/nutrition` - Log new meal
- `GET /api/nutrition/{id}` - Get meal by ID
- `PUT /api/nutrition/{id}` - Update meal
- `DELETE /api/nutrition/{id}` - Delete meal
- `POST /api/nutrition/ai-plan` - Generate AI meal plan

### Progress Tracking
- `GET /api/progress` - Get user progress
- `POST /api/progress` - Log progress entry
- `GET /api/progress/analytics` - Get progress analytics
- `GET /api/progress/insights` - Get AI insights

### Chatbot
- `POST /api/chatbot` - Send message to chatbot
- `GET /api/chatbot/history` - Get chat history
- `DELETE /api/chatbot/history/{id}` - Delete chat session

### Form Check
- `POST /api/pose/check` - Upload video/image for form analysis
- `GET /api/pose/history` - Get form check history

## 🔒 Security

- JWT-based authentication
- Password encryption using BCrypt
- Role-based access control (RBAC)
- CORS configuration for frontend integration
- Input validation and sanitization
- Rate limiting (to be implemented)

## 🚀 Deployment

### Deploy to Render

1. **Push to GitHub**:
```bash
git add .
git commit -m "Initial backend setup"
git push origin main
```

2. **Connect to Render**:
- Go to [Render Dashboard](https://dashboard.render.com)
- Create new Web Service
- Connect your GitHub repository
- Use the provided `render.yaml` configuration

3. **Configure environment variables** in Render dashboard:
- `MONGODB_URI`
- `JWT_SECRET`
- `CORS_ORIGINS`
- AI service URLs

### Manual Docker Deployment

1. **Build Docker image**:
```bash
docker build -t fitness-backend .
```

2. **Run container**:
```bash
docker run -p 8080:8080 \
  -e MONGODB_URI=your-mongodb-uri \
  -e JWT_SECRET=your-secret \
  fitness-backend
```

## 🧪 Testing

Run tests:
```bash
mvn test
```

Run with coverage:
```bash
mvn test jacoco:report
```

## 📊 Monitoring

- Health check: `/actuator/health`
- Metrics: `/actuator/metrics`
- Info: `/actuator/info`

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License.

## 🏗️ Project Structure

```
backend/
├── src/
│   ├── main/
│   │   ├── java/com/aifitness/backend/
│   │   │   ├── config/         # Configuration classes
│   │   │   ├── controller/     # REST controllers
│   │   │   ├── dto/           # Data Transfer Objects
│   │   │   ├── entity/        # MongoDB entities
│   │   │   ├── exception/     # Custom exceptions
│   │   │   ├── mapper/        # Object mappers
│   │   │   ├── repository/    # MongoDB repositories
│   │   │   ├── security/      # Security configuration
│   │   │   ├── service/       # Business logic
│   │   │   ├── util/          # Utility classes
│   │   │   └── validation/    # Custom validators
│   │   └── resources/
│   │       └── application.yml # Application configuration
│   └── test/                  # Test classes
├── Dockerfile                  # Docker configuration
├── docker-compose.yml         # Docker Compose setup
├── pom.xml                    # Maven dependencies
├── render.yaml                # Render deployment config
└── README.md                  # Documentation
```

## 🐛 Troubleshooting

### MongoDB Connection Issues
- Ensure MongoDB is running: `docker ps | grep mongo`
- Check connection string format
- Verify network connectivity

### JWT Token Issues
- Ensure JWT secret is properly set
- Check token expiration settings
- Verify token format in requests

### CORS Issues
- Update `CORS_ORIGINS` environment variable
- Check allowed methods and headers

## 📞 Support

For issues and questions:
- Create an issue in the GitHub repository
- Check existing issues for solutions
- Review API documentation

## 🔄 Next Steps

1. Implement email service for notifications
2. Add rate limiting for API endpoints
3. Implement file storage (S3 integration)
4. Add comprehensive unit and integration tests
5. Set up CI/CD pipeline
6. Implement WebSocket for real-time features
7. Add API versioning
8. Implement audit logging#   A I t h l e t e _ b a c k e n d  
 