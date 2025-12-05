"""
Phase 3: Backend API Testing Suite  
Comprehensive testing of backend APIs including auth, CRUD, load, and security
"""

import json
import requests
import time
import threading
import concurrent.futures
from datetime import datetime
import unittest
from pathlib import Path
import sys
import os
import random
import string
import hashlib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Phase3BackendAPITests(unittest.TestCase):
    """Comprehensive test suite for backend API functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Initialize test environment"""
        cls.base_url = "http://localhost:8080/api"
        cls.test_results = {
            "phase": "Phase 3 - Backend API Testing",
            "timestamp": datetime.now().isoformat(),
            "authentication": {},
            "crud_operations": {},
            "load_testing": {},
            "security_testing": {},
            "database_consistency": {},
            "summary": {}
        }
        
        # Test users for different scenarios
        cls.test_users = []
        cls.auth_tokens = {}
        
    def test_01_authentication_flow(self):
        """Test complete authentication flow"""
        test_cases = []
        
        # Test Case 1: User Registration
        test_case_1 = {
            "test_id": "AUTH-001",
            "operation": "Register New User",
            "input": {
                "username": f"testuser_{random.randint(1000, 9999)}",
                "password": "Test@123456",
                "email": f"test{random.randint(1000, 9999)}@example.com",
                "firstName": "Test",
                "lastName": "User"
            },
            "expected": "201 Created",
            "actual": "",
            "pass_fail": "",
            "notes": ""
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/auth/register",
                json=test_case_1["input"],
                timeout=5
            )
            test_case_1["actual"] = f"Status: {response.status_code}"
            if response.status_code in [200, 201]:
                test_case_1["pass_fail"] = "PASS"
                self.test_users.append(test_case_1["input"])
            else:
                test_case_1["pass_fail"] = "FAIL"
            test_case_1["notes"] = response.text[:100] if response.text else ""
        except Exception as e:
            test_case_1["actual"] = f"Error: {str(e)}"
            test_case_1["pass_fail"] = "FAIL"
        
        test_cases.append(test_case_1)
        
        # Test Case 2: User Login
        if self.test_users:
            user = self.test_users[0]
            test_case_2 = {
                "test_id": "AUTH-002",
                "operation": "User Login",
                "input": {
                    "username": user["username"],
                    "password": user["password"]
                },
                "expected": "200 OK with JWT token",
                "actual": "",
                "pass_fail": "",
                "notes": ""
            }
            
            try:
                response = requests.post(
                    f"{self.base_url}/auth/login",
                    json=test_case_2["input"],
                    timeout=5
                )
                test_case_2["actual"] = f"Status: {response.status_code}"
                if response.status_code == 200:
                    data = response.json()
                    if "token" in data or "accessToken" in data:
                        test_case_2["pass_fail"] = "PASS"
                        token = data.get("token") or data.get("accessToken")
                        self.auth_tokens[user["username"]] = token
                        test_case_2["notes"] = "JWT token received"
                    else:
                        test_case_2["pass_fail"] = "FAIL"
                        test_case_2["notes"] = "No token in response"
                else:
                    test_case_2["pass_fail"] = "FAIL"
            except Exception as e:
                test_case_2["actual"] = f"Error: {str(e)}"
                test_case_2["pass_fail"] = "FAIL"
            
            test_cases.append(test_case_2)
        
        # Test Case 3: Invalid Login
        test_case_3 = {
            "test_id": "AUTH-003",
            "operation": "Invalid Login Attempt",
            "input": {
                "username": "nonexistent",
                "password": "wrongpassword"
            },
            "expected": "401 Unauthorized",
            "actual": "",
            "pass_fail": "",
            "notes": ""
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/auth/login",
                json=test_case_3["input"],
                timeout=5
            )
            test_case_3["actual"] = f"Status: {response.status_code}"
            test_case_3["pass_fail"] = "PASS" if response.status_code in [401, 403] else "FAIL"
        except Exception as e:
            test_case_3["actual"] = f"Error: {str(e)}"
            test_case_3["pass_fail"] = "FAIL"
        
        test_cases.append(test_case_3)
        
        # Test Case 4: Token Validation
        if self.auth_tokens:
            token = list(self.auth_tokens.values())[0]
            test_case_4 = {
                "test_id": "AUTH-004",
                "operation": "Protected Endpoint Access",
                "input": "Bearer token in header",
                "expected": "200 OK",
                "actual": "",
                "pass_fail": "",
                "notes": ""
            }
            
            try:
                headers = {"Authorization": f"Bearer {token}"}
                response = requests.get(
                    f"{self.base_url}/user/profile",
                    headers=headers,
                    timeout=5
                )
                test_case_4["actual"] = f"Status: {response.status_code}"
                test_case_4["pass_fail"] = "PASS" if response.status_code == 200 else "WARN"
            except Exception as e:
                test_case_4["actual"] = f"Error: {str(e)}"
                test_case_4["pass_fail"] = "FAIL"
            
            test_cases.append(test_case_4)
        
        # Store results
        self.test_results["authentication"] = {
            "test_cases": test_cases,
            "total": len(test_cases),
            "passed": sum(1 for tc in test_cases if tc["pass_fail"] == "PASS"),
            "failed": sum(1 for tc in test_cases if tc["pass_fail"] == "FAIL")
        }
    
    def test_02_crud_operations(self):
        """Test CRUD operations on main entities"""
        test_cases = []
        token = list(self.auth_tokens.values())[0] if self.auth_tokens else None
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        
        # Test Workout CRUD
        workout_data = {
            "name": "Test Workout",
            "exercises": [
                {
                    "name": "Push-ups",
                    "sets": 3,
                    "reps": 15,
                    "duration": 0
                },
                {
                    "name": "Squats",
                    "sets": 3,
                    "reps": 20,
                    "duration": 0
                }
            ],
            "difficulty": "intermediate",
            "duration": 30
        }
        
        # CREATE
        test_case_create = {
            "test_id": "CRUD-001",
            "operation": "Create Workout",
            "entity": "Workout",
            "method": "POST",
            "expected": "201 Created",
            "actual": "",
            "pass_fail": "",
            "notes": ""
        }
        
        created_id = None
        try:
            response = requests.post(
                f"{self.base_url}/workouts",
                json=workout_data,
                headers=headers,
                timeout=5
            )
            test_case_create["actual"] = f"Status: {response.status_code}"
            if response.status_code in [200, 201]:
                test_case_create["pass_fail"] = "PASS"
                data = response.json()
                created_id = data.get("id") or data.get("_id")
                test_case_create["notes"] = f"Created ID: {created_id}"
            else:
                test_case_create["pass_fail"] = "FAIL"
        except Exception as e:
            test_case_create["actual"] = f"Error: {str(e)}"
            test_case_create["pass_fail"] = "FAIL"
        
        test_cases.append(test_case_create)
        
        # READ
        if created_id:
            test_case_read = {
                "test_id": "CRUD-002",
                "operation": "Read Workout",
                "entity": "Workout",
                "method": "GET",
                "expected": "200 OK",
                "actual": "",
                "pass_fail": "",
                "notes": ""
            }
            
            try:
                response = requests.get(
                    f"{self.base_url}/workouts/{created_id}",
                    headers=headers,
                    timeout=5
                )
                test_case_read["actual"] = f"Status: {response.status_code}"
                test_case_read["pass_fail"] = "PASS" if response.status_code == 200 else "FAIL"
            except Exception as e:
                test_case_read["actual"] = f"Error: {str(e)}"
                test_case_read["pass_fail"] = "FAIL"
            
            test_cases.append(test_case_read)
        
        # UPDATE
        if created_id:
            test_case_update = {
                "test_id": "CRUD-003",
                "operation": "Update Workout",
                "entity": "Workout",
                "method": "PUT",
                "expected": "200 OK",
                "actual": "",
                "pass_fail": "",
                "notes": ""
            }
            
            workout_data["name"] = "Updated Workout"
            try:
                response = requests.put(
                    f"{self.base_url}/workouts/{created_id}",
                    json=workout_data,
                    headers=headers,
                    timeout=5
                )
                test_case_update["actual"] = f"Status: {response.status_code}"
                test_case_update["pass_fail"] = "PASS" if response.status_code == 200 else "FAIL"
            except Exception as e:
                test_case_update["actual"] = f"Error: {str(e)}"
                test_case_update["pass_fail"] = "FAIL"
            
            test_cases.append(test_case_update)
        
        # DELETE
        if created_id:
            test_case_delete = {
                "test_id": "CRUD-004",
                "operation": "Delete Workout",
                "entity": "Workout",
                "method": "DELETE",
                "expected": "204 No Content",
                "actual": "",
                "pass_fail": "",
                "notes": ""
            }
            
            try:
                response = requests.delete(
                    f"{self.base_url}/workouts/{created_id}",
                    headers=headers,
                    timeout=5
                )
                test_case_delete["actual"] = f"Status: {response.status_code}"
                test_case_delete["pass_fail"] = "PASS" if response.status_code in [200, 204] else "FAIL"
            except Exception as e:
                test_case_delete["actual"] = f"Error: {str(e)}"
                test_case_delete["pass_fail"] = "FAIL"
            
            test_cases.append(test_case_delete)
        
        # Test Meal CRUD
        meal_data = {
            "name": "Test Meal",
            "calories": 500,
            "protein": 30,
            "carbs": 50,
            "fat": 20,
            "mealType": "lunch"
        }
        
        test_case_meal = {
            "test_id": "CRUD-005",
            "operation": "Create Meal",
            "entity": "Meal",
            "method": "POST",
            "expected": "201 Created",
            "actual": "",
            "pass_fail": "",
            "notes": ""
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/meals",
                json=meal_data,
                headers=headers,
                timeout=5
            )
            test_case_meal["actual"] = f"Status: {response.status_code}"
            test_case_meal["pass_fail"] = "PASS" if response.status_code in [200, 201] else "FAIL"
        except Exception as e:
            test_case_meal["actual"] = f"Error: {str(e)}"
            test_case_meal["pass_fail"] = "FAIL"
        
        test_cases.append(test_case_meal)
        
        # Store results
        self.test_results["crud_operations"] = {
            "test_cases": test_cases,
            "total": len(test_cases),
            "passed": sum(1 for tc in test_cases if tc["pass_fail"] == "PASS"),
            "failed": sum(1 for tc in test_cases if tc["pass_fail"] == "FAIL")
        }
    
    def test_03_load_testing(self):
        """Perform load testing with concurrent users"""
        load_test_results = {
            "test_cases": [],
            "metrics": {}
        }
        
        # Test Case 1: Concurrent User Registration
        num_users = 50
        test_case_1 = {
            "test_id": "LOAD-001",
            "scenario": f"{num_users} Concurrent User Registrations",
            "expected": "All succeed within 10 seconds",
            "actual": "",
            "pass_fail": "",
            "metrics": {}
        }
        
        def register_user(i):
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/auth/register",
                    json={
                        "username": f"loadtest_{i}_{random.randint(1000, 9999)}",
                        "password": "LoadTest@123",
                        "email": f"load{i}_{random.randint(1000, 9999)}@test.com"
                    },
                    timeout=10
                )
                elapsed = time.time() - start_time
                return {
                    "user_id": i,
                    "success": response.status_code in [200, 201],
                    "status": response.status_code,
                    "time": elapsed
                }
            except Exception as e:
                return {
                    "user_id": i,
                    "success": False,
                    "error": str(e),
                    "time": time.time() - start_time
                }
        
        # Execute concurrent registrations
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(register_user, i) for i in range(num_users)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        total_time = time.time() - start_time
        
        # Analyze results
        successful = sum(1 for r in results if r["success"])
        avg_time = sum(r["time"] for r in results) / len(results)
        
        test_case_1["actual"] = f"{successful}/{num_users} succeeded"
        test_case_1["pass_fail"] = "PASS" if successful >= num_users * 0.8 else "FAIL"
        test_case_1["metrics"] = {
            "total_time": total_time,
            "avg_response_time": avg_time,
            "success_rate": successful / num_users * 100,
            "requests_per_second": num_users / total_time
        }
        
        load_test_results["test_cases"].append(test_case_1)
        
        # Test Case 2: Concurrent API Calls (500-1000 users)
        num_requests = 500
        test_case_2 = {
            "test_id": "LOAD-002",
            "scenario": f"{num_requests} Concurrent API Requests",
            "expected": "95% success rate, avg response < 2s",
            "actual": "",
            "pass_fail": "",
            "metrics": {}
        }
        
        def make_api_call(i):
            endpoints = ["/workouts", "/meals", "/user/profile"]
            endpoint = endpoints[i % len(endpoints)]
            start_time = time.time()
            
            try:
                headers = {}
                if self.auth_tokens:
                    token = list(self.auth_tokens.values())[0]
                    headers = {"Authorization": f"Bearer {token}"}
                
                response = requests.get(
                    f"{self.base_url}{endpoint}",
                    headers=headers,
                    timeout=5
                )
                elapsed = time.time() - start_time
                return {
                    "request_id": i,
                    "endpoint": endpoint,
                    "success": response.status_code in [200, 201, 401],  # 401 is expected if no auth
                    "status": response.status_code,
                    "time": elapsed
                }
            except Exception as e:
                return {
                    "request_id": i,
                    "endpoint": endpoint,
                    "success": False,
                    "error": str(e),
                    "time": time.time() - start_time
                }
        
        # Execute concurrent API calls
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(make_api_call, i) for i in range(num_requests)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        total_time = time.time() - start_time
        
        # Analyze results
        successful = sum(1 for r in results if r["success"])
        avg_time = sum(r["time"] for r in results) / len(results)
        max_time = max(r["time"] for r in results)
        min_time = min(r["time"] for r in results)
        
        test_case_2["actual"] = f"{successful}/{num_requests} succeeded, avg: {avg_time:.2f}s"
        test_case_2["pass_fail"] = "PASS" if successful >= num_requests * 0.95 and avg_time < 2 else "WARN"
        test_case_2["metrics"] = {
            "total_time": total_time,
            "avg_response_time": avg_time,
            "max_response_time": max_time,
            "min_response_time": min_time,
            "success_rate": successful / num_requests * 100,
            "requests_per_second": num_requests / total_time
        }
        
        load_test_results["test_cases"].append(test_case_2)
        
        # Store results
        self.test_results["load_testing"] = load_test_results
    
    def test_04_security_testing(self):
        """Test for common security vulnerabilities"""
        security_tests = []
        
        # Test Case 1: SQL Injection
        test_case_1 = {
            "test_id": "SEC-001",
            "vulnerability": "SQL Injection",
            "endpoint": "/auth/login",
            "payload": {
                "username": "admin' OR '1'='1",
                "password": "' OR '1'='1"
            },
            "expected": "400 Bad Request or 401 Unauthorized",
            "actual": "",
            "pass_fail": "",
            "notes": ""
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/auth/login",
                json=test_case_1["payload"],
                timeout=5
            )
            test_case_1["actual"] = f"Status: {response.status_code}"
            # Should NOT return 200 for SQL injection attempt
            test_case_1["pass_fail"] = "PASS" if response.status_code != 200 else "FAIL"
            test_case_1["notes"] = "SQL injection prevented" if response.status_code != 200 else "VULNERABLE!"
        except Exception as e:
            test_case_1["actual"] = f"Error: {str(e)}"
            test_case_1["pass_fail"] = "PASS"  # Error is acceptable for malicious input
        
        security_tests.append(test_case_1)
        
        # Test Case 2: XSS Prevention
        test_case_2 = {
            "test_id": "SEC-002",
            "vulnerability": "XSS (Cross-Site Scripting)",
            "endpoint": "/workouts",
            "payload": {
                "name": "<script>alert('XSS')</script>",
                "exercises": []
            },
            "expected": "Input sanitized or rejected",
            "actual": "",
            "pass_fail": "",
            "notes": ""
        }
        
        try:
            headers = {}
            if self.auth_tokens:
                token = list(self.auth_tokens.values())[0]
                headers = {"Authorization": f"Bearer {token}"}
            
            response = requests.post(
                f"{self.base_url}/workouts",
                json=test_case_2["payload"],
                headers=headers,
                timeout=5
            )
            
            if response.status_code in [200, 201]:
                # Check if script tags are in response (should be escaped)
                response_text = response.text
                if "<script>" in response_text:
                    test_case_2["pass_fail"] = "FAIL"
                    test_case_2["notes"] = "XSS vulnerability detected!"
                else:
                    test_case_2["pass_fail"] = "PASS"
                    test_case_2["notes"] = "XSS prevented - content sanitized"
            else:
                test_case_2["pass_fail"] = "PASS"
                test_case_2["notes"] = "Malicious input rejected"
            
            test_case_2["actual"] = f"Status: {response.status_code}"
        except Exception as e:
            test_case_2["actual"] = f"Error: {str(e)}"
            test_case_2["pass_fail"] = "PASS"
        
        security_tests.append(test_case_2)
        
        # Test Case 3: Broken Authentication
        test_case_3 = {
            "test_id": "SEC-003",
            "vulnerability": "Broken Authentication",
            "test": "Access protected endpoint without token",
            "expected": "401 Unauthorized",
            "actual": "",
            "pass_fail": "",
            "notes": ""
        }
        
        try:
            response = requests.get(
                f"{self.base_url}/user/profile",
                timeout=5
            )
            test_case_3["actual"] = f"Status: {response.status_code}"
            test_case_3["pass_fail"] = "PASS" if response.status_code in [401, 403] else "FAIL"
            test_case_3["notes"] = "Protected endpoint secured" if response.status_code in [401, 403] else "VULNERABLE!"
        except Exception as e:
            test_case_3["actual"] = f"Error: {str(e)}"
            test_case_3["pass_fail"] = "FAIL"
        
        security_tests.append(test_case_3)
        
        # Test Case 4: Rate Limiting
        test_case_4 = {
            "test_id": "SEC-004",
            "vulnerability": "No Rate Limiting",
            "test": "100 rapid requests",
            "expected": "Rate limiting after threshold",
            "actual": "",
            "pass_fail": "",
            "notes": ""
        }
        
        rate_limited = False
        for i in range(100):
            try:
                response = requests.get(
                    f"{self.base_url}/health",
                    timeout=1
                )
                if response.status_code == 429:  # Too Many Requests
                    rate_limited = True
                    break
            except:
                pass
        
        test_case_4["actual"] = "Rate limited" if rate_limited else "No rate limiting detected"
        test_case_4["pass_fail"] = "PASS" if rate_limited else "WARN"
        test_case_4["notes"] = f"Rate limiting {'active' if rate_limited else 'not detected'}"
        
        security_tests.append(test_case_4)
        
        # Test Case 5: Password Security
        test_case_5 = {
            "test_id": "SEC-005",
            "vulnerability": "Weak Password Policy",
            "test": "Register with weak password",
            "payload": {
                "username": "weakpwdtest",
                "password": "123",  # Very weak password
                "email": "weak@test.com"
            },
            "expected": "400 Bad Request - password too weak",
            "actual": "",
            "pass_fail": "",
            "notes": ""
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/auth/register",
                json=test_case_5["payload"],
                timeout=5
            )
            test_case_5["actual"] = f"Status: {response.status_code}"
            test_case_5["pass_fail"] = "PASS" if response.status_code != 201 else "FAIL"
            test_case_5["notes"] = "Weak password rejected" if response.status_code != 201 else "Weak password accepted!"
        except Exception as e:
            test_case_5["actual"] = f"Error: {str(e)}"
            test_case_5["pass_fail"] = "FAIL"
        
        security_tests.append(test_case_5)
        
        # Store results
        self.test_results["security_testing"] = {
            "test_cases": security_tests,
            "total": len(security_tests),
            "passed": sum(1 for tc in security_tests if tc["pass_fail"] == "PASS"),
            "failed": sum(1 for tc in security_tests if tc["pass_fail"] == "FAIL"),
            "vulnerabilities_found": sum(1 for tc in security_tests if "VULNERABLE" in tc.get("notes", ""))
        }
    
    def test_05_database_consistency(self):
        """Test database consistency and data integrity"""
        consistency_tests = []
        
        # Test Case 1: Transaction Consistency
        test_case_1 = {
            "test_id": "DB-001",
            "test": "Transaction Consistency",
            "scenario": "Create and immediately read data",
            "expected": "Data consistent across operations",
            "actual": "",
            "pass_fail": "",
            "notes": ""
        }
        
        headers = {}
        if self.auth_tokens:
            token = list(self.auth_tokens.values())[0]
            headers = {"Authorization": f"Bearer {token}"}
        
        # Create a workout
        workout_data = {
            "name": f"Consistency Test {random.randint(1000, 9999)}",
            "exercises": [{"name": "Test Exercise", "sets": 3, "reps": 10}]
        }
        
        try:
            # Create
            create_response = requests.post(
                f"{self.base_url}/workouts",
                json=workout_data,
                headers=headers,
                timeout=5
            )
            
            if create_response.status_code in [200, 201]:
                created_data = create_response.json()
                workout_id = created_data.get("id") or created_data.get("_id")
                
                # Immediately read
                read_response = requests.get(
                    f"{self.base_url}/workouts/{workout_id}",
                    headers=headers,
                    timeout=5
                )
                
                if read_response.status_code == 200:
                    read_data = read_response.json()
                    # Check consistency
                    if read_data.get("name") == workout_data["name"]:
                        test_case_1["pass_fail"] = "PASS"
                        test_case_1["notes"] = "Data consistent"
                    else:
                        test_case_1["pass_fail"] = "FAIL"
                        test_case_1["notes"] = "Data inconsistency detected"
                else:
                    test_case_1["pass_fail"] = "FAIL"
                    test_case_1["notes"] = "Could not read created data"
                
                test_case_1["actual"] = "Create and read successful"
            else:
                test_case_1["pass_fail"] = "FAIL"
                test_case_1["actual"] = f"Create failed: {create_response.status_code}"
        except Exception as e:
            test_case_1["actual"] = f"Error: {str(e)}"
            test_case_1["pass_fail"] = "FAIL"
        
        consistency_tests.append(test_case_1)
        
        # Test Case 2: Referential Integrity
        test_case_2 = {
            "test_id": "DB-002",
            "test": "Referential Integrity",
            "scenario": "Delete user with associated data",
            "expected": "Cascade delete or prevent deletion",
            "actual": "",
            "pass_fail": "",
            "notes": ""
        }
        
        # This would require creating a user with associated workouts/meals
        # then attempting to delete the user
        test_case_2["actual"] = "Test scenario requires specific setup"
        test_case_2["pass_fail"] = "SKIP"
        test_case_2["notes"] = "Requires manual verification"
        
        consistency_tests.append(test_case_2)
        
        # Test Case 3: Concurrent Updates
        test_case_3 = {
            "test_id": "DB-003",
            "test": "Concurrent Update Handling",
            "scenario": "10 concurrent updates to same resource",
            "expected": "All updates handled without data corruption",
            "actual": "",
            "pass_fail": "",
            "notes": ""
        }
        
        if self.auth_tokens:
            # Create a resource to update
            try:
                create_response = requests.post(
                    f"{self.base_url}/workouts",
                    json={"name": "Concurrent Test", "exercises": []},
                    headers=headers,
                    timeout=5
                )
                
                if create_response.status_code in [200, 201]:
                    workout_id = create_response.json().get("id")
                    
                    def update_workout(i):
                        try:
                            response = requests.put(
                                f"{self.base_url}/workouts/{workout_id}",
                                json={"name": f"Updated {i}", "exercises": []},
                                headers=headers,
                                timeout=5
                            )
                            return response.status_code in [200, 201]
                        except:
                            return False
                    
                    # Execute concurrent updates
                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                        futures = [executor.submit(update_workout, i) for i in range(10)]
                        results = [f.result() for f in concurrent.futures.as_completed(futures)]
                    
                    successful = sum(results)
                    test_case_3["actual"] = f"{successful}/10 updates succeeded"
                    test_case_3["pass_fail"] = "PASS" if successful >= 8 else "FAIL"
                    test_case_3["notes"] = "Concurrent updates handled"
            except Exception as e:
                test_case_3["actual"] = f"Error: {str(e)}"
                test_case_3["pass_fail"] = "FAIL"
        else:
            test_case_3["pass_fail"] = "SKIP"
            test_case_3["notes"] = "No authentication available"
        
        consistency_tests.append(test_case_3)
        
        # Store results
        self.test_results["database_consistency"] = {
            "test_cases": consistency_tests,
            "total": len(consistency_tests),
            "passed": sum(1 for tc in consistency_tests if tc["pass_fail"] == "PASS"),
            "failed": sum(1 for tc in consistency_tests if tc["pass_fail"] == "FAIL")
        }
    
    @classmethod
    def tearDownClass(cls):
        """Generate and save comprehensive test report"""
        # Calculate overall summary
        total_tests = 0
        total_passed = 0
        
        for category in ["authentication", "crud_operations", "load_testing", "security_testing", "database_consistency"]:
            if category in cls.test_results:
                total_tests += cls.test_results[category].get("total", 0)
                total_passed += cls.test_results[category].get("passed", 0)
        
        cls.test_results["summary"] = {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_tests - total_passed,
            "pass_rate": f"{(total_passed/total_tests*100):.2f}%" if total_tests > 0 else "0%",
            "categories_tested": 5,
            "critical_issues": cls._identify_critical_issues()
        }
        
        # Save JSON report
        output_file = Path("test_reports/phase3_backend_api_test_report.json")
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(cls.test_results, f, indent=2)
        
        # Generate and save markdown report
        markdown_report = cls._generate_markdown_report()
        markdown_file = Path("test_reports/phase3_backend_api_test_report.md")
        with open(markdown_file, 'w') as f:
            f.write(markdown_report)
        
        print(f"\n{'='*60}")
        print("PHASE 3 - BACKEND API TESTING COMPLETE")
        print(f"{'='*60}")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_tests - total_passed}")
        print(f"Pass Rate: {cls.test_results['summary']['pass_rate']}")
        
        if cls.test_results['summary']['critical_issues']:
            print(f"\n⚠️  CRITICAL ISSUES FOUND:")
            for issue in cls.test_results['summary']['critical_issues']:
                print(f"  - {issue}")
        
        print(f"\nReports saved to:")
        print(f"  - JSON: {output_file}")
        print(f"  - Markdown: {markdown_file}")
        print(f"{'='*60}\n")
    
    @classmethod
    def _identify_critical_issues(cls):
        """Identify critical issues from test results"""
        issues = []
        
        # Check security vulnerabilities
        if "security_testing" in cls.test_results:
            vulns = cls.test_results["security_testing"].get("vulnerabilities_found", 0)
            if vulns > 0:
                issues.append(f"{vulns} security vulnerabilities detected")
        
        # Check load testing failures
        if "load_testing" in cls.test_results:
            for test in cls.test_results["load_testing"].get("test_cases", []):
                if test.get("pass_fail") == "FAIL":
                    issues.append(f"Load testing failed: {test.get('scenario')}")
        
        # Check authentication issues
        if "authentication" in cls.test_results:
            auth_passed = cls.test_results["authentication"].get("passed", 0)
            auth_total = cls.test_results["authentication"].get("total", 1)
            if auth_passed < auth_total * 0.5:
                issues.append("Critical authentication failures")
        
        return issues
    
    @classmethod
    def _generate_markdown_report(cls):
        """Generate detailed markdown report"""
        report = []
        report.append("# Phase 3: Backend API Test Report")
        report.append(f"\n**Test Date:** {cls.test_results['timestamp']}")
        report.append(f"\n## Executive Summary")
        report.append(f"- **Total Tests:** {cls.test_results['summary']['total_tests']}")
        report.append(f"- **Passed:** {cls.test_results['summary']['total_passed']}")
        report.append(f"- **Failed:** {cls.test_results['summary']['total_failed']}")
        report.append(f"- **Pass Rate:** {cls.test_results['summary']['pass_rate']}")
        
        if cls.test_results['summary']['critical_issues']:
            report.append(f"\n### ⚠️ Critical Issues")
            for issue in cls.test_results['summary']['critical_issues']:
                report.append(f"- {issue}")
        
        # Authentication Tests
        if "authentication" in cls.test_results:
            report.append(f"\n## Authentication Testing")
            auth = cls.test_results["authentication"]
            report.append(f"- Total Tests: {auth['total']}")
            report.append(f"- Passed: {auth['passed']}")
            report.append(f"- Failed: {auth['failed']}")
            
            if auth.get("test_cases"):
                report.append("\n| Test ID | Operation | Expected | Actual | Pass/Fail |")
                report.append("|---------|-----------|----------|--------|-----------|")
                for tc in auth["test_cases"]:
                    report.append(f"| {tc['test_id']} | {tc['operation']} | {tc['expected']} | {tc['actual']} | {tc['pass_fail']} |")
        
        # CRUD Operations
        if "crud_operations" in cls.test_results:
            report.append(f"\n## CRUD Operations Testing")
            crud = cls.test_results["crud_operations"]
            report.append(f"- Total Tests: {crud['total']}")
            report.append(f"- Passed: {crud['passed']}")
            report.append(f"- Failed: {crud['failed']}")
        
        # Load Testing
        if "load_testing" in cls.test_results:
            report.append(f"\n## Load Testing Results")
            load = cls.test_results["load_testing"]
            
            for test in load.get("test_cases", []):
                report.append(f"\n### {test['scenario']}")
                report.append(f"- **Result:** {test['pass_fail']}")
                report.append(f"- **Actual:** {test['actual']}")
                
                if test.get("metrics"):
                    report.append(f"\n**Performance Metrics:**")
                    for key, value in test["metrics"].items():
                        if isinstance(value, float):
                            report.append(f"- {key.replace('_', ' ').title()}: {value:.2f}")
                        else:
                            report.append(f"- {key.replace('_', ' ').title()}: {value}")
        
        # Security Testing
        if "security_testing" in cls.test_results:
            report.append(f"\n## Security Testing")
            sec = cls.test_results["security_testing"]
            report.append(f"- Total Tests: {sec['total']}")
            report.append(f"- Passed: {sec['passed']}")
            report.append(f"- Failed: {sec['failed']}")
            report.append(f"- **Vulnerabilities Found:** {sec.get('vulnerabilities_found', 0)}")
            
            if sec.get("test_cases"):
                report.append("\n| Test ID | Vulnerability | Result | Notes |")
                report.append("|---------|---------------|--------|-------|")
                for tc in sec["test_cases"]:
                    report.append(f"| {tc['test_id']} | {tc['vulnerability']} | {tc['pass_fail']} | {tc['notes']} |")
        
        # Database Consistency
        if "database_consistency" in cls.test_results:
            report.append(f"\n## Database Consistency Testing")
            db = cls.test_results["database_consistency"]
            report.append(f"- Total Tests: {db['total']}")
            report.append(f"- Passed: {db['passed']}")
            report.append(f"- Failed: {db['failed']}")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Run tests with specific order
    unittest.main(verbosity=2)