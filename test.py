import unittest
import pandas as pd
import numpy as np
import json
from unittest.mock import patch, MagicMock
import os
import sys
import streamlit as st
from io import StringIO
from contextlib import redirect_stdout

# Import the app module - assuming the main code is in a file called app.py
# You might need to adjust this import based on how your project is structured
sys.path.append('.')
import app  # Import your Streamlit app here

class TestBusinessValuationMethods(unittest.TestCase):
    """Test the core business valuation methods"""
    
    def test_safe_float(self):
        """Test the safe_float helper function"""
        self.assertEqual(app.safe_float("$1,234.56"), 1234.56)
        self.assertEqual(app.safe_float("invalid"), 0.0)
        self.assertEqual(app.safe_float(None), 0.0)
        self.assertEqual(app.safe_float(None, default=1.0), 1.0)
    
    def test_safe_int(self):
        """Test the safe_int helper function"""
        self.assertEqual(app.safe_int("$1,234.56"), 1234)
        self.assertEqual(app.safe_int("invalid"), 0)
        self.assertEqual(app.safe_int(None), 0)
        self.assertEqual(app.safe_int(None, default=1), 1)
    
    def test_calculate_valuation(self):
        """Test the calculate_valuation function with different scenarios"""
        # Test with revenue only
        company_data = {
            "name": "Test Company",
            "industry": "Software/SaaS",
            "revenue": 1000000,
            "growth": "High",
            "earnings": None,
            "assets": None,
            "liabilities": None,
            "cash_flows": []
        }
        results = app.calculate_valuation(company_data)
        self.assertIn('revenue_valuation', results)
        self.assertEqual(results['revenue_valuation'], 10000000)  # 10x multiple for High growth SaaS
        
        # Test with revenue and earnings
        company_data["earnings"] = 200000
        results = app.calculate_valuation(company_data)
        self.assertIn('earnings_valuation', results)
        self.assertEqual(results['earnings_valuation'], 5000000)  # 25x PE ratio for SaaS
        
        # Test with assets and liabilities
        company_data["assets"] = 5000000
        company_data["liabilities"] = 2000000
        results = app.calculate_valuation(company_data)
        self.assertIn('asset_based_valuation', results)
        self.assertEqual(results['asset_based_valuation'], 3000000)
        
        # Test with cash flows
        company_data["cash_flows"] = [300000, 350000, 400000, 450000, 500000]
        results = app.calculate_valuation(company_data)
        self.assertIn('dcf_valuation', results)
        self.assertGreater(results['dcf_valuation'], 0)
        
        # Test with different industry
        company_data["industry"] = "E-commerce"
        results = app.calculate_valuation(company_data)
        self.assertNotEqual(results['revenue_valuation'], 10000000)  # Should use different multiple


class TestLLMFunctions(unittest.TestCase):
    """Test the LLM integration functions"""
    
    @patch('app.gemini_model.generate_content')
    def test_gemini_qna(self, mock_generate):
        """Test the Gemini Q&A function"""
        # Mock the response from Gemini
        mock_response = MagicMock()
        mock_response.text = "This is a test response"
        mock_generate.return_value = mock_response
        
        # Test the function
        result = app.gemini_qna("Test query", "Test context")
        self.assertEqual(result, "This is a test response")
        mock_generate.assert_called_once()
    
    @patch('app.gemini_qna')
    def test_generate_next_question(self, mock_qna):
        """Test the generate_next_question function"""
        # Mock the database query and Gemini response
        app.questions_collection = MagicMock()
        app.questions_collection.find_one.return_value = {"question": "Database question?"}
        mock_qna.return_value = "Generated question?"
        
        # Test with industry
        result = app.generate_next_question([], industry="Software")
        self.assertEqual(result, "Database question?")
        
        # Test without industry or when no DB result
        app.questions_collection.find_one.return_value = None
        result = app.generate_next_question([{"question": "Q1", "answer": "A1"}])
        self.assertEqual(result, "Generated question?")


class TestMongoIntegration(unittest.TestCase):
    """Test MongoDB connection and query functions"""
    
    @patch('app.MongoClient')
    def test_get_mongo_client(self, mock_mongo):
        """Test the MongoDB client connection with retry logic"""
        mock_client = MagicMock()
        mock_mongo.return_value = mock_client
        
        result = app.get_mongo_client()
        self.assertEqual(result, mock_client)
        
        # Test retry logic by simulating failure then success
        mock_mongo.side_effect = [Exception("Connection failed"), mock_client]
        with patch('app.time.sleep') as mock_sleep:
            result = app.get_mongo_client()
            self.assertEqual(result, mock_client)
            mock_sleep.assert_called_once()
    
    @patch('app.listings_collection.find')
    def test_get_similar_businesses(self, mock_find):
        """Test the similar businesses query function"""
        # Mock the MongoDB query result
        mock_cursor = MagicMock()
        mock_cursor.limit.return_value = [{"business_basics": {"business_name": "Test Biz"}}]
        mock_find.return_value = mock_cursor
        
        result = app.get_similar_businesses("Software", app.listings_collection)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["business_basics"]["business_name"], "Test Biz")


class TestStreamlitIntegration(unittest.TestCase):
    """Test Streamlit integration - these tests will be more challenging and may require mocking Streamlit"""
    
    def setUp(self):
        """Set up test environment for Streamlit"""
        # Mock Streamlit session state
        if not hasattr(st, 'session_state'):
            st.session_state = {}
        
        # Initialize session state variables
        st.session_state.valuation_data = {}
        st.session_state.assessment_responses = {}
        st.session_state.conversation_history = []
        st.session_state.current_question_idx = 0
        st.session_state.assessment_completed = False
    
    @patch('app.st.form')
    @patch('app.st.form_submit_button')
    @patch('app.calculate_valuation')
    @patch('app.get_similar_businesses')
    def test_valuation_form_submission(self, mock_get_similar, mock_calculate, mock_submit, mock_form):
        """Test the valuation form submission flow"""
        # Mock form inputs
        mock_form.return_value.__enter__.return_value = None
        mock_submit.return_value = True
        
        # Mock calculation results
        mock_calculate.return_value = {"revenue_valuation": 1000000}
        mock_get_similar.return_value = []
        
        # Create test input data
        # Here we would need to mock all the st.text_input, st.number_input etc.
        # This is complex and might require a custom testing framework for Streamlit
        # For demonstration purposes, we'll just verify the function calls
        
        # Call the render function directly (might need adjustments)
        try:
            app.render_valuation_page()
        except Exception as e:
            # Expected to fail in tests due to complex Streamlit mocking needs
            pass
        
        # Check if the calculation functions were called
        mock_calculate.assert_called()
        mock_get_similar.assert_called()


class TestEndToEndScenarios(unittest.TestCase):
    """End-to-end scenario tests with simulated data"""
    
    def test_saas_company_valuation(self):
        """Test a complete SaaS company valuation scenario"""
        # Define test input
        company_data = {
            "name": "SaaSCo",
            "industry": "Software/SaaS",
            "revenue": 2000000,
            "earnings": 500000,
            "assets": 1000000,
            "liabilities": 200000,
            "growth": "High",
            "cash_flows": [600000, 800000, 1000000, 1200000, 1500000]
        }
        
        # Calculate valuation
        results = app.calculate_valuation(company_data)
        
        # Check all valuation methods are present
        self.assertIn('revenue_valuation', results)
        self.assertIn('earnings_valuation', results)
        self.assertIn('asset_based_valuation', results)
        self.assertIn('dcf_valuation', results)
        
        # Validate the results against expected ranges
        # Revenue valuation: 10x multiple for High growth SaaS
        self.assertEqual(results['revenue_valuation'], 20000000)
        
        # Earnings valuation: 25x PE for SaaS
        self.assertEqual(results['earnings_valuation'], 12500000)
        
        # Asset-based: assets - liabilities
        self.assertEqual(results['asset_based_valuation'], 800000)
        
        # DCF should be positive and reasonable
        self.assertGreater(results['dcf_valuation'], 0)
        
        # Average valuation should be within reasonable bounds
        avg_valuation = sum(results.values()) / len(results)
        self.assertGreater(avg_valuation, min(results.values()))
        self.assertLess(avg_valuation, max(results.values()))


class TestInputValidation(unittest.TestCase):
    """Test input validation functions"""
    
    def test_validate_input(self):
        """Test the validate_input function"""
        # Test string validation
        valid, _ = app.validate_input("Test", "string", min_value=3, max_value=10)
        self.assertTrue(valid)
        
        valid, error = app.validate_input("Te", "string", min_value=3)
        self.assertFalse(valid)
        self.assertIn("at least 3 characters", error)
        
        # Test numeric validation
        valid, _ = app.validate_input("100", "float", min_value=0, max_value=1000)
        self.assertTrue(valid)
        
        valid, error = app.validate_input("2000", "float", max_value=1000)
        self.assertFalse(valid)
        self.assertIn("less than 1000", error)
        
        valid, error = app.validate_input("abc", "float")
        self.assertFalse(valid)
        self.assertIn("Invalid float input", error)


class TestFormattingFunctions(unittest.TestCase):
    """Test formatting functions"""
    
    def test_format_currency(self):
        """Test the format_currency function"""
        self.assertEqual(app.format_currency(1234.56), "$1,234.56")
        self.assertEqual(app.format_currency("1234.56"), "$1,234.56")
        self.assertEqual(app.format_currency(None), "$0.00")
        self.assertEqual(app.format_currency("invalid"), "$0.00")


# Test data generator for realistic test scenarios
def generate_test_data():
    """Generate realistic test data for valuation and assessment tests"""
    industries = ["Software/SaaS", "E-commerce", "Technology", "Healthcare", "Food & Beverage"]
    growth_rates = ["High", "Moderate", "Low"]
    
    test_companies = []
    
    # Generate 5 test companies with realistic data
    for i in range(5):
        revenue_base = np.random.randint(500000, 10000000)
        margin = np.random.uniform(0.1, 0.3)
        growth_factor = {"High": 1.5, "Moderate": 1.3, "Low": 1.1}
        
        industry = np.random.choice(industries)
        growth = np.random.choice(growth_rates)
        
        earnings = revenue_base * margin
        assets = revenue_base * np.random.uniform(0.5, 2.0)
        liabilities = assets * np.random.uniform(0.2, 0.6)
        
        # Generate realistic cash flows
        cash_flows = []
        for year in range(5):
            cf = earnings * (growth_factor[growth] ** year) * np.random.uniform(0.9, 1.1)
            cash_flows.append(cf)
        
        company = {
            "name": f"TestCo {i+1}",
            "industry": industry,
            "revenue": revenue_base,
            "earnings": earnings,
            "assets": assets,
            "liabilities": liabilities,
            "growth": growth,
            "cash_flows": cash_flows
        }
        
        test_companies.append(company)
    
    return test_companies


# Main test runner
if __name__ == "__main__":
    # Generate test data first
    test_data = generate_test_data()
    with open('test_data.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"Generated test data for {len(test_data)} companies")
    
    # Run the tests
    unittest.main()
