from google import genai
import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import time
import random

class GeminiRateLimiter:
    def __init__(self, max_requests=180, period=60, min_interval=0.1):
        """
        Initialize the rate limiter.
        
        Args:
            max_requests (int): Maximum number of requests allowed in the period (default: 180).
            period (int): Time period in seconds (default: 60 for 1 minute).
            min_interval (float): Minimum time in seconds between consecutive requests (default: 0.1).
        """
        self.max_requests = max_requests
        self.period = period  # in seconds
        self.min_interval = min_interval  # minimum seconds between requests
        self.request_times = []  # List to store timestamps of requests
        self.last_request_time = 0  # Timestamp of the last request
        self.lock = asyncio.Lock()  # Ensure thread-safe updates
        
    async def acquire(self):
        """
        Acquires permission to make a request, waiting if necessary to respect rate limits.
        """
        async with self.lock:
            now = time.time()
            
            # Enforce minimum time between requests
            time_since_last_request = now - self.last_request_time
            if time_since_last_request < self.min_interval:
                await asyncio.sleep(self.min_interval - time_since_last_request)
                now = time.time()  # Update current time after sleeping
            
            # Remove timestamps older than the period
            cutoff = now - self.period
            self.request_times = [t for t in self.request_times if t > cutoff]
            
            # Wait if at maximum requests for the period
            if len(self.request_times) >= self.max_requests:
                oldest = self.request_times[0]
                sleep_time = oldest + self.period - now
                if sleep_time > 0:
                    # Add a small random offset to prevent potential thundering herd issues
                    sleep_time += random.uniform(0.1, 0.5)
                    await asyncio.sleep(sleep_time)
                    now = time.time()  # Update current time after sleeping
                    # Recalculate after sleeping
                    cutoff = now - self.period
                    self.request_times = [t for t in self.request_times if t > cutoff]
            
            # Record this request
            self.request_times.append(now)
            self.last_request_time = now
            
    def get_quota_usage(self):
        """
        Returns current usage information.
        
        Returns:
            dict: Dictionary with usage information.
        """
        now = time.time()
        cutoff = now - self.period
        self.request_times = [t for t in self.request_times if t > cutoff]
        
        return {
            "requests_in_period": len(self.request_times),
            "max_requests": self.max_requests,
            "usage_percent": (len(self.request_times) / self.max_requests) * 100 if self.max_requests > 0 else 0,
            "period_seconds": self.period
        }


# Initialize global rate limiter
gemini_limiter = GeminiRateLimiter()


async def clean_text_with_gemini_async(text: str, max_retries=5, initial_delay=4) -> str:
    """
    Clean and format text using Gemini AI.
    
    Args:
        text (str): Text to clean
        max_retries (int): Maximum number of retries on failure
        initial_delay (int): Initial delay for exponential backoff
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
        
    # Limit text size
    if len(text) > 100000:
        text = text[:100000]
        
    global failed_cleaning_requests
    
    # Truncate text if it's excessively long
    text = text[:50000] if len(text) > 50000 else text
    
    retry_count = 0
    delay = initial_delay
    
    while retry_count < max_retries:
        try:
            # Wait for rate limiter
            await gemini_limiter.acquire()
            
            model = genai.GenerativeModel('gemini-pro')
            prompt = f"""
            Please clean and format the following SEC filing text to make it more readable and remove HTML artifacts, page numbers, etc.
            Return ONLY the cleaned text with NO additional commentary or explanation.
            TEXT: {text}
            """
            
            response = model.generate_content(prompt)
            cleaned_text = response.text.strip()
            
            # Check if text was actually cleaned
            if not cleaned_text or cleaned_text.startswith("I'm unable to"):
                raise ValueError("Failed to clean text properly")
                
            return cleaned_text
            
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"Failed to clean text after {max_retries} attempts: {e}")
                return text  # Return original text as fallback
                
            print(f"Retrying text cleaning (attempt {retry_count}/{max_retries}): {e}")
            await asyncio.sleep(delay)
            delay *= 2  # Exponential backoff
            
    return text  # Fallback to original text


async def create_tags_with_gemini_async(text: str, section_name: str, symbol: str, max_retries=5, initial_delay=4) -> list:
    """
    Generate topic tags for a section of text using Gemini AI.
    
    Args:
        text (str): Text to analyze
        section_name (str): Name of the document section
        symbol (str): Company ticker symbol
        max_retries (int): Maximum number of retries on failure
        initial_delay (int): Initial delay for exponential backoff
        
    Returns:
        list: List of tags
    """
    if not text:
        return []
        
    # Limit text size
    if len(text) > 100000:
        text = text[:100000]
        
    global failed_tagging_requests
    
    # Truncate text if it's excessively long
    text = text[:30000] if len(text) > 30000 else text
    
    retry_count = 0
    delay = initial_delay
    
    while retry_count < max_retries:
        try:
            # Wait for rate limiter
            await gemini_limiter.acquire()
            
            model = genai.GenerativeModel('gemini-pro')
            prompt = f"""
            You are analyzing a section of an SEC 10-K filing for company {symbol}.
            
            Create a JSON array of tags representing the key topics, risks, and themes in this text.
            These tags should be short phrases (1-3 words) that highlight important information.
            Focus on specific, meaningful topics rather than generic tags.
            Include 10-20 tags depending on the content.
            
            Section name: {section_name}
            Text: {text}
            
            Return ONLY a valid JSON array of strings with no additional text or explanation.
            """
            
            response = model.generate_content(prompt)
            
            # Parse the JSON response
            # Handle potential response formats
            response_text = response.text.strip()
            
            # If response includes markdown code block formatting, extract just the JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
                
            # Remove any explanatory text before or after the JSON array
            if response_text.find("[") > 0:
                response_text = response_text[response_text.find("["):]
            if response_text.rfind("]") < len(response_text) - 1:
                response_text = response_text[:response_text.rfind("]") + 1]
                
            tags = json.loads(response_text)
            
            if not isinstance(tags, list):
                raise ValueError("Response is not a list")
                
            return tags
            
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"Failed to create tags after {max_retries} attempts: {e}")
                return []  # Return empty list as fallback
                
            print(f"Retrying tag creation (attempt {retry_count}/{max_retries}): {e}")
            await asyncio.sleep(delay)
            delay *= 2  # Exponential backoff
            
    return []  # Fallback to empty list 