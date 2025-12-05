"""
Phase 2: Backend AI Integration Testing Suite
Tests all backend endpoints that integrate with AI services
"""

import json
import requests
import time
import concurrent.futures
from datetime import datetime
import unittest
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Phase2BackendAIIntegrationTests(unittest.TestCase):
    """Comprehensive test suite for backend AI service integration"""
    
    @classmethod
    def setUpClass(cls):
        """Initialize test environment"""
        cls.base_url = "http://localhost:8080/api"
        cls.test_results = {
            "phase": "Phase 2 - Backend AI Integration Testing",
            "timestamp": datetime.now().isoformat(),
            "endpoints": {},
            "performance_metrics": {},
            "error_handling": {},
            "summary": {}
        }
        
        # Test user credentials
        cls.test_user = {
            "email": "test@aitest.com",
            "password": "Test123!",
            "name": "AI Test User"
        }
        
        cls.auth_token = None
        
        # Try to get auth token
        try:
            cls._setup_auth()
        except Exception as e:
            print(f"Warning: Could not setup authentication: {e}")
    
    @classmethod
    def _setup_auth(cls):
        """Setup authentication for protected endpoints"""
        # First try to register
        register_response = requests.post(
            f"{cls.base_url}/auth/register",
            json=cls.test_user
        )
        
        # Then login
        login_response = requests.post(
            f"{cls.base_url}/auth/login",
            json={
                "email": cls.test_user["email"],
                "password": cls.test_user["password"]
            }
        )
        
        if login_response.status_code == 200:
            cls.auth_token = login_response.json().get("token")
    
    def test_pose_check_endpoint(self):
        """Test /pose-check endpoint with various inputs"""
        endpoint = "/pose-check"
        test_cases = []
        
        # Test Case 1: Valid pose data
        test_case_1 = {
            "test_id": "PC-API-001",
            "endpoint": endpoint,
            "input": {
                "image_data": "base64_encoded_image_here",
                "exercise_type": "squat",
                "keypoints": [[0.5, 0.5, 0.9]] * 17  # Mock keypoints
            },
            "expected_response": {
                "status": 200,
                "has_feedback": True,
                "has_confidence_score": True
            },
            "actual_response": {},
            "pass_fail": "",
            "latency_ms": 0
        }
        
        start_time = time.time()
        try:
            headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
            response = requests.post(
                f"{self.base_url}{endpoint}",
                json=test_case_1["input"],
                headers=headers,
                timeout=10
            )
            test_case_1["latency_ms"] = (time.time() - start_time) * 1000
            test_case_1["actual_response"] = {
                "status": response.status_code,
                "body": response.json() if response.status_code == 200 else response.text
            }
            
            # Validate response
            if response.status_code == 200:
                body = response.json()
                has_feedback = "feedback" in body or "result" in body
                has_confidence = "confidence" in body or "score" in body
                test_case_1["pass_fail"] = "PASS" if has_feedback else "FAIL"
            else:
                test_case_1["pass_fail"] = "FAIL"
        except Exception as e:
            test_case_1["actual_response"] = {"error": str(e)}
            test_case_1["pass_fail"] = "FAIL"
            test_case_1["latency_ms"] = (time.time() - start_time) * 1000
        
        test_cases.append(test_case_1)
        
        # Test Case 2: Invalid input handling
        test_case_2 = {
            "test_id": "PC-API-002",
            "endpoint": endpoint,
            "input": {
                "invalid_field": "test"
            },
            "expected_response": {
                "status": 400,
                "has_error_message": True
            },
            "actual_response": {},
            "pass_fail": "",
            "latency_ms": 0
        }
        
        start_time = time.time()
        try:
            headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
            response = requests.post(
                f"{self.base_url}{endpoint}",
                json=test_case_2["input"],
                headers=headers,
                timeout=10
            )
            test_case_2["latency_ms"] = (time.time() - start_time) * 1000
            test_case_2["actual_response"] = {
                "status": response.status_code,
                "body": response.json() if response.content else response.text
            }
            
            # Should return 400 for invalid input
            test_case_2["pass_fail"] = "PASS" if response.status_code in [400, 422] else "FAIL"
        except Exception as e:
            test_case_2["actual_response"] = {"error": str(e)}
            test_case_2["pass_fail"] = "FAIL"
        
        test_cases.append(test_case_2)
        
        # Test Case 3: Large payload handling
        test_case_3 = {
            "test_id": "PC-API-003",
            "endpoint": endpoint,
            "input": {
                "image_data": "x" * 1000000,  # 1MB string
                "exercise_type": "deadlift",
                "keypoints": [[0.5, 0.5, 0.9]] * 17
            },
            "expected_response": {
                "status_ok": True,
                "timeout_handling": True
            },
            "actual_response": {},
            "pass_fail": "",
            "latency_ms": 0
        }
        
        start_time = time.time()
        try:
            headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
            response = requests.post(
                f"{self.base_url}{endpoint}",
                json=test_case_3["input"],
                headers=headers,
                timeout=30
            )
            test_case_3["latency_ms"] = (time.time() - start_time) * 1000
            test_case_3["actual_response"] = {
                "status": response.status_code,
                "handled": True
            }
            test_case_3["pass_fail"] = "PASS" if response.status_code in [200, 413, 400] else "FAIL"
        except requests.Timeout:
            test_case_3["actual_response"] = {"error": "Timeout"}
            test_case_3["pass_fail"] = "WARN"
        except Exception as e:
            test_case_3["actual_response"] = {"error": str(e)}
            test_case_3["pass_fail"] = "FAIL"
        
        test_cases.append(test_case_3)
        
        # Store results
        self.test_results["endpoints"]["pose_check"] = {
            "test_cases": test_cases,
            "total": len(test_cases),
            "passed": sum(1 for tc in test_cases if tc["pass_fail"] == "PASS"),
            "failed": sum(1 for tc in test_cases if tc["pass_fail"] == "FAIL"),
            "avg_latency": sum(tc["latency_ms"] for tc in test_cases) / len(test_cases)
        }
    
    def test_workout_recommendation_endpoint(self):
        """Test /recommend-workout endpoint"""
        endpoint = "/recommend-workout"
        test_cases = []
        
        # Test Case 1: Beginner profile
        test_case_1 = {
            "test_id": "WR-API-001",
            "endpoint": endpoint,
            "input": {
                "user_profile": {
                    "age": 25,
                    "fitness_level": "beginner",
                    "goals": ["weight_loss"],
                    "equipment": ["none"],
                    "injuries": []
                }
            },
            "expected_response": {
                "status": 200,
                "has_workout_plan": True
            },
            "actual_response": {},
            "pass_fail": "",
            "latency_ms": 0
        }
        
        start_time = time.time()
        try:
            headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
            response = requests.post(
                f"{self.base_url}{endpoint}",
                json=test_case_1["input"],
                headers=headers,
                timeout=10
            )
            test_case_1["latency_ms"] = (time.time() - start_time) * 1000
            test_case_1["actual_response"] = {
                "status": response.status_code,
                "body": response.json() if response.status_code == 200 else response.text
            }
            
            if response.status_code == 200:
                body = response.json()
                has_plan = "workout_plan" in body or "exercises" in body or "recommendations" in body
                test_case_1["pass_fail"] = "PASS" if has_plan else "FAIL"
            else:
                test_case_1["pass_fail"] = "FAIL"
        except Exception as e:
            test_case_1["actual_response"] = {"error": str(e)}
            test_case_1["pass_fail"] = "FAIL"
        
        test_cases.append(test_case_1)
        
        # Test Case 2: Advanced profile with injuries
        test_case_2 = {
            "test_id": "WR-API-002",
            "endpoint": endpoint,
            "input": {
                "user_profile": {
                    "age": 35,
                    "fitness_level": "advanced",
                    "goals": ["muscle_gain", "strength"],
                    "equipment": ["barbell", "dumbbells"],
                    "injuries": ["knee", "lower_back"]
                }
            },
            "expected_response": {
                "status": 200,
                "considers_injuries": True
            },
            "actual_response": {},
            "pass_fail": "",
            "latency_ms": 0
        }
        
        start_time = time.time()
        try:
            headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
            response = requests.post(
                f"{self.base_url}{endpoint}",
                json=test_case_2["input"],
                headers=headers,
                timeout=10
            )
            test_case_2["latency_ms"] = (time.time() - start_time) * 1000
            test_case_2["actual_response"] = {
                "status": response.status_code,
                "body": response.json() if response.status_code == 200 else response.text
            }
            
            if response.status_code == 200:
                body = response.json()
                # Check if response considers injuries
                body_str = json.dumps(body).lower()
                avoids_injuries = "knee" not in body_str or "modification" in body_str
                test_case_2["pass_fail"] = "PASS" if avoids_injuries else "WARN"
            else:
                test_case_2["pass_fail"] = "FAIL"
        except Exception as e:
            test_case_2["actual_response"] = {"error": str(e)}
            test_case_2["pass_fail"] = "FAIL"
        
        test_cases.append(test_case_2)
        
        # Store results
        self.test_results["endpoints"]["workout_recommendation"] = {
            "test_cases": test_cases,
            "total": len(test_cases),
            "passed": sum(1 for tc in test_cases if tc["pass_fail"] == "PASS"),
            "failed": sum(1 for tc in test_cases if tc["pass_fail"] == "FAIL"),
            "avg_latency": sum(tc["latency_ms"] for tc in test_cases) / len(test_cases) if test_cases else 0
        }
    
    def test_nutrition_plan_endpoint(self):
        """Test /nutrition-plan endpoint"""
        endpoint = "/nutrition-plan"
        test_cases = []
        
        # Test Case 1: Vegan diet with allergies
        test_case_1 = {
            "test_id": "NP-API-001",
            "endpoint": endpoint,
            "input": {
                "dietary_preferences": {
                    "diet_type": "vegan",
                    "allergies": ["nuts", "soy"],
                    "calorie_target": 2000,
                    "meal_count": 3
                }
            },
            "expected_response": {
                "status": 200,
                "respects_preferences": True
            },
            "actual_response": {},
            "pass_fail": "",
            "latency_ms": 0
        }
        
        start_time = time.time()
        try:
            headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
            response = requests.post(
                f"{self.base_url}{endpoint}",
                json=test_case_1["input"],
                headers=headers,
                timeout=10
            )
            test_case_1["latency_ms"] = (time.time() - start_time) * 1000
            test_case_1["actual_response"] = {
                "status": response.status_code,
                "body": response.json() if response.status_code == 200 else response.text
            }
            
            if response.status_code == 200:
                body = response.json()
                body_str = json.dumps(body).lower()
                # Check if allergens are avoided
                no_allergens = "nut" not in body_str and "soy" not in body_str
                test_case_1["pass_fail"] = "PASS" if no_allergens else "FAIL"
            else:
                test_case_1["pass_fail"] = "FAIL"
        except Exception as e:
            test_case_1["actual_response"] = {"error": str(e)}
            test_case_1["pass_fail"] = "FAIL"
        
        test_cases.append(test_case_1)
        
        # Test Case 2: Keto diet
        test_case_2 = {
            "test_id": "NP-API-002",
            "endpoint": endpoint,
            "input": {
                "dietary_preferences": {
                    "diet_type": "keto",
                    "calorie_target": 1800,
                    "macros": {
                        "carbs": 5,
                        "protein": 25,
                        "fat": 70
                    }
                }
            },
            "expected_response": {
                "status": 200,
                "follows_macros": True
            },
            "actual_response": {},
            "pass_fail": "",
            "latency_ms": 0
        }
        
        start_time = time.time()
        try:
            headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
            response = requests.post(
                f"{self.base_url}{endpoint}",
                json=test_case_2["input"],
                headers=headers,
                timeout=10
            )
            test_case_2["latency_ms"] = (time.time() - start_time) * 1000
            test_case_2["actual_response"] = {
                "status": response.status_code,
                "body": response.json() if response.status_code == 200 else response.text
            }
            
            test_case_2["pass_fail"] = "PASS" if response.status_code == 200 else "FAIL"
        except Exception as e:
            test_case_2["actual_response"] = {"error": str(e)}
            test_case_2["pass_fail"] = "FAIL"
        
        test_cases.append(test_case_2)
        
        # Store results
        self.test_results["endpoints"]["nutrition_plan"] = {
            "test_cases": test_cases,
            "total": len(test_cases),
            "passed": sum(1 for tc in test_cases if tc["pass_fail"] == "PASS"),
            "failed": sum(1 for tc in test_cases if tc["pass_fail"] == "FAIL"),
            "avg_latency": sum(tc["latency_ms"] for tc in test_cases) / len(test_cases) if test_cases else 0
        }
    
    def test_chatbot_endpoint(self):
        """Test /chatbot endpoint"""
        endpoint = "/chatbot"
        test_cases = []
        
        # Test Case 1: General fitness question
        test_case_1 = {
            "test_id": "CB-API-001",
            "endpoint": endpoint,
            "input": {
                "message": "What's the best exercise for abs?",
                "context": {
                    "user_level": "intermediate"
                }
            },
            "expected_response": {
                "status": 200,
                "has_response": True
            },
            "actual_response": {},
            "pass_fail": "",
            "latency_ms": 0
        }
        
        start_time = time.time()
        try:
            headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
            response = requests.post(
                f"{self.base_url}{endpoint}",
                json=test_case_1["input"],
                headers=headers,
                timeout=10
            )
            test_case_1["latency_ms"] = (time.time() - start_time) * 1000
            test_case_1["actual_response"] = {
                "status": response.status_code,
                "body": response.json() if response.status_code == 200 else response.text
            }
            
            if response.status_code == 200:
                body = response.json()
                has_response = "response" in body or "message" in body or "answer" in body
                test_case_1["pass_fail"] = "PASS" if has_response else "FAIL"
            else:
                test_case_1["pass_fail"] = "FAIL"
        except Exception as e:
            test_case_1["actual_response"] = {"error": str(e)}
            test_case_1["pass_fail"] = "FAIL"
        
        test_cases.append(test_case_1)
        
        # Test Case 2: Medical query (should be handled safely)
        test_case_2 = {
            "test_id": "CB-API-002",
            "endpoint": endpoint,
            "input": {
                "message": "I have severe chest pain during exercise",
                "context": {}
            },
            "expected_response": {
                "status": 200,
                "safe_response": True
            },
            "actual_response": {},
            "pass_fail": "",
            "latency_ms": 0
        }
        
        start_time = time.time()
        try:
            headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
            response = requests.post(
                f"{self.base_url}{endpoint}",
                json=test_case_2["input"],
                headers=headers,
                timeout=10
            )
            test_case_2["latency_ms"] = (time.time() - start_time) * 1000
            test_case_2["actual_response"] = {
                "status": response.status_code,
                "body": response.json() if response.status_code == 200 else response.text
            }
            
            if response.status_code == 200:
                body = response.json()
                body_str = json.dumps(body).lower()
                # Should recommend medical consultation
                safe_response = "doctor" in body_str or "medical" in body_str or "professional" in body_str
                test_case_2["pass_fail"] = "PASS" if safe_response else "FAIL"
            else:
                test_case_2["pass_fail"] = "FAIL"
        except Exception as e:
            test_case_2["actual_response"] = {"error": str(e)}
            test_case_2["pass_fail"] = "FAIL"
        
        test_cases.append(test_case_2)
        
        # Store results
        self.test_results["endpoints"]["chatbot"] = {
            "test_cases": test_cases,
            "total": len(test_cases),
            "passed": sum(1 for tc in test_cases if tc["pass_fail"] == "PASS"),
            "failed": sum(1 for tc in test_cases if tc["pass_fail"] == "FAIL"),
            "avg_latency": sum(tc["latency_ms"] for tc in test_cases) / len(test_cases) if test_cases else 0
        }
    
    def test_error_handling(self):
        """Test error handling for all endpoints"""
        error_cases = []
        
        # Test timeout handling
        endpoints = ["/pose-check", "/recommend-workout", "/nutrition-plan", "/chatbot"]
        
        for endpoint in endpoints:
            test_case = {
                "test_id": f"ERR-{endpoint.replace('/', '').upper()}-001",
                "endpoint": endpoint,
                "test_type": "timeout",
                "input": {"delay_simulation": 30},
                "expected": "Graceful timeout handling",
                "actual": "",
                "pass_fail": ""
            }
            
            try:
                headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
                response = requests.post(
                    f"{self.base_url}{endpoint}",
                    json=test_case["input"],
                    headers=headers,
                    timeout=1  # Very short timeout
                )
                test_case["actual"] = "No timeout occurred"
                test_case["pass_fail"] = "FAIL"
            except requests.Timeout:
                test_case["actual"] = "Timeout handled gracefully"
                test_case["pass_fail"] = "PASS"
            except Exception as e:
                test_case["actual"] = f"Unexpected error: {str(e)}"
                test_case["pass_fail"] = "FAIL"
            
            error_cases.append(test_case)
        
        # Test invalid JSON
        for endpoint in endpoints:
            test_case = {
                "test_id": f"ERR-{endpoint.replace('/', '').upper()}-002",
                "endpoint": endpoint,
                "test_type": "invalid_json",
                "expected": "400 Bad Request",
                "actual": "",
                "pass_fail": ""
            }
            
            try:
                headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
                headers["Content-Type"] = "application/json"
                response = requests.post(
                    f"{self.base_url}{endpoint}",
                    data="invalid json {{}",  # Malformed JSON
                    headers=headers,
                    timeout=5
                )
                test_case["actual"] = f"Status: {response.status_code}"
                test_case["pass_fail"] = "PASS" if response.status_code in [400, 422] else "FAIL"
            except Exception as e:
                test_case["actual"] = f"Exception: {str(e)}"
                test_case["pass_fail"] = "FAIL"
            
            error_cases.append(test_case)
        
        self.test_results["error_handling"] = {
            "test_cases": error_cases,
            "total": len(error_cases),
            "passed": sum(1 for tc in error_cases if tc["pass_fail"] == "PASS"),
            "failed": sum(1 for tc in error_cases if tc["pass_fail"] == "FAIL")
        }
    
    def test_load_and_performance(self):
        """Test load handling and performance metrics"""
        performance_results = {
            "concurrent_requests": [],
            "response_times": [],
            "error_rate": 0
        }
        
        # Test concurrent requests
        num_concurrent = 10
        endpoint = "/chatbot"  # Test with chatbot as it's likely the most complex
        
        def make_request(i):
            try:
                start = time.time()
                headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
                response = requests.post(
                    f"{self.base_url}{endpoint}",
                    json={"message": f"Test message {i}"},
                    headers=headers,
                    timeout=10
                )
                elapsed = time.time() - start
                return {
                    "request_id": i,
                    "status": response.status_code,
                    "time": elapsed,
                    "success": response.status_code == 200
                }
            except Exception as e:
                return {
                    "request_id": i,
                    "status": 0,
                    "time": 10,
                    "success": False,
                    "error": str(e)
                }
        
        # Execute concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_concurrent)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # Analyze results
        successful_requests = sum(1 for r in results if r["success"])
        avg_response_time = sum(r["time"] for r in results) / len(results)
        max_response_time = max(r["time"] for r in results)
        min_response_time = min(r["time"] for r in results)
        
        performance_results["concurrent_requests"] = results
        performance_results["response_times"] = [r["time"] for r in results]
        performance_results["error_rate"] = (num_concurrent - successful_requests) / num_concurrent
        performance_results["metrics"] = {
            "total_requests": num_concurrent,
            "successful": successful_requests,
            "failed": num_concurrent - successful_requests,
            "avg_response_time_ms": avg_response_time * 1000,
            "max_response_time_ms": max_response_time * 1000,
            "min_response_time_ms": min_response_time * 1000,
            "error_rate_percent": performance_results["error_rate"] * 100
        }
        
        self.test_results["performance_metrics"] = performance_results
    
    @classmethod
    def tearDownClass(cls):
        """Save test results and generate report"""
        # Calculate overall summary
        total_tests = 0
        total_passed = 0
        
        for endpoint_name, endpoint_results in cls.test_results["endpoints"].items():
            total_tests += endpoint_results["total"]
            total_passed += endpoint_results["passed"]
        
        if "error_handling" in cls.test_results:
            total_tests += cls.test_results["error_handling"]["total"]
            total_passed += cls.test_results["error_handling"]["passed"]
        
        cls.test_results["summary"] = {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_tests - total_passed,
            "pass_rate": f"{(total_passed/total_tests*100):.2f}%" if total_tests > 0 else "0%",
            "endpoints_tested": len(cls.test_results["endpoints"]),
            "avg_latency_ms": sum(
                ep.get("avg_latency", 0) 
                for ep in cls.test_results["endpoints"].values()
            ) / len(cls.test_results["endpoints"]) if cls.test_results["endpoints"] else 0
        }
        
        # Save detailed JSON report
        output_file = Path("test_reports/phase2_backend_ai_integration_report.json")
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(cls.test_results, f, indent=2)
        
        # Generate markdown report
        markdown_report = cls._generate_markdown_report()
        markdown_file = Path("test_reports/phase2_backend_ai_integration_report.md")
        with open(markdown_file, 'w') as f:
            f.write(markdown_report)
        
        print(f"\n{'='*60}")
        print("PHASE 2 - BACKEND AI INTEGRATION TESTING COMPLETE")
        print(f"{'='*60}")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_tests - total_passed}")
        print(f"Pass Rate: {cls.test_results['summary']['pass_rate']}")
        print(f"Average Latency: {cls.test_results['summary']['avg_latency_ms']:.2f}ms")
        print(f"\nReports saved to:")
        print(f"  - JSON: {output_file}")
        print(f"  - Markdown: {markdown_file}")
        print(f"{'='*60}\n")
    
    @classmethod
    def _generate_markdown_report(cls):
        """Generate a markdown formatted report"""
        report = []
        report.append("# Phase 2: Backend AI Integration Test Report")
        report.append(f"\n**Test Date:** {cls.test_results['timestamp']}")
        report.append(f"\n## Summary")
        report.append(f"- **Total Tests:** {cls.test_results['summary']['total_tests']}")
        report.append(f"- **Passed:** {cls.test_results['summary']['total_passed']}")
        report.append(f"- **Failed:** {cls.test_results['summary']['total_failed']}")
        report.append(f"- **Pass Rate:** {cls.test_results['summary']['pass_rate']}")
        report.append(f"- **Average Latency:** {cls.test_results['summary']['avg_latency_ms']:.2f}ms\n")
        
        # Endpoint results
        report.append("## Endpoint Test Results\n")
        for endpoint_name, results in cls.test_results["endpoints"].items():
            report.append(f"### {endpoint_name.replace('_', ' ').title()}")
            report.append(f"- Tests: {results['total']}")
            report.append(f"- Passed: {results['passed']}")
            report.append(f"- Failed: {results['failed']}")
            report.append(f"- Average Latency: {results.get('avg_latency', 0):.2f}ms\n")
            
            # Test cases table
            if "test_cases" in results and results["test_cases"]:
                report.append("| Test ID | Input | Expected | Actual | Pass/Fail | Latency |")
                report.append("|---------|-------|----------|--------|-----------|---------|")
                for tc in results["test_cases"]:
                    input_summary = str(tc.get("input", ""))[:30] + "..."
                    expected = str(tc.get("expected_response", ""))[:30] + "..."
                    actual = str(tc.get("actual_response", ""))[:30] + "..."
                    report.append(f"| {tc['test_id']} | {input_summary} | {expected} | {actual} | {tc['pass_fail']} | {tc.get('latency_ms', 0):.0f}ms |")
                report.append("")
        
        # Error handling results
        if "error_handling" in cls.test_results:
            report.append("## Error Handling Tests\n")
            results = cls.test_results["error_handling"]
            report.append(f"- Total Tests: {results['total']}")
            report.append(f"- Passed: {results['passed']}")
            report.append(f"- Failed: {results['failed']}\n")
        
        # Performance metrics
        if "performance_metrics" in cls.test_results and "metrics" in cls.test_results["performance_metrics"]:
            metrics = cls.test_results["performance_metrics"]["metrics"]
            report.append("## Performance Metrics\n")
            report.append(f"- **Concurrent Requests:** {metrics['total_requests']}")
            report.append(f"- **Successful:** {metrics['successful']}")
            report.append(f"- **Failed:** {metrics['failed']}")
            report.append(f"- **Error Rate:** {metrics['error_rate_percent']:.2f}%")
            report.append(f"- **Avg Response Time:** {metrics['avg_response_time_ms']:.2f}ms")
            report.append(f"- **Max Response Time:** {metrics['max_response_time_ms']:.2f}ms")
            report.append(f"- **Min Response Time:** {metrics['min_response_time_ms']:.2f}ms\n")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Check if backend is running
    try:
        response = requests.get("http://localhost:8080/actuator/health", timeout=2)
        if response.status_code != 200:
            print("WARNING: Backend may not be running properly")
    except:
        print("WARNING: Cannot connect to backend at localhost:8080")
        print("Please ensure the backend is running before executing tests")
    
    # Run the tests
    unittest.main(verbosity=2)