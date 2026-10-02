"""
Google Gemini AI Integration for News Credibility Analysis
Provides real-time fact-checking suggestions and source credibility insights
"""
import os
from typing import Dict, Any, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


class GeminiInsightService:
    """
    Lightweight Gemini API wrapper for news article analysis.
    Uses gemini-3.5-flash-lite for cost-efficient inference.
    """
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.enabled = bool(self.api_key and self.api_key != "your_gemini_api_key_here")
        self.client = genai.Client(api_key=self.api_key) if self.enabled else None
        # Use minimal / fast model
        self.model_name = "gemini-3.5-flash-lite"

    def analyze_news_article(self, title: str, text: str, prediction: str, confidence: float) -> Dict[str, Any]:
        """
        Generate AI-powered credibility insights for a news article.
        
        Args:
            title: Article headline
            text: Article body text
            prediction: ML model prediction (FAKE/REAL)
            confidence: ML model confidence score
        
        Returns:
            Dictionary containing Gemini insights or fallback message
        """
        if not self.enabled or not self.client:
            return {
                "enabled": False,
                "message": "Gemini API key not configured. Add GEMINI_API_KEY to your .env file.",
                "credibilityScore": None,
                "factCheckSuggestions": [],
                "sourceAnalysis": "",
                "redFlags": [],
                "verificationTips": []
            }
        
        try:
            prompt = self._build_analysis_prompt(title, text, prediction, confidence)
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    top_p=0.8,
                    max_output_tokens=512,
                )
            )
            
            return self._parse_gemini_response(response.text or "")
        
        except Exception as e:
            return {
                "enabled": True,
                "error": str(e),
                "message": f"Gemini API error: {str(e)[:120]}",
                "credibilityScore": None,
                "factCheckSuggestions": [],
                "sourceAnalysis": "",
                "redFlags": [],
                "verificationTips": []
            }
    
    def _build_analysis_prompt(self, title: str, text: str, prediction: str, confidence: float) -> str:
        """Construct structured prompt for Gemini analysis"""
        truncated_text = text[:800] if len(text) > 800 else text
        
        return f"""Analyze this news article for credibility indicators. Our ML model classified it as {prediction} with {confidence:.1%} confidence.

Title: {title}

Text: {truncated_text}

Provide a concise analysis in this exact format:

CREDIBILITY_SCORE: [0-100 integer score]
RED_FLAGS: [comma-separated list of specific warning signs, or "None detected"]
FACT_CHECK: [2-3 specific claims that should be verified]
SOURCE_ANALYSIS: [1 sentence about likely source type and reliability indicators]
VERIFICATION_TIPS: [2-3 actionable steps to verify this article]

Keep responses factual and concise. Focus on linguistic patterns, claim verifiability, and journalistic standards."""
    
    def _parse_gemini_response(self, response_text: str) -> Dict[str, any]:
        """Parse structured Gemini response into JSON-friendly format"""
        lines = response_text.strip().split("\n")
        parsed = {
            "enabled": True,
            "credibilityScore": None,
            "factCheckSuggestions": [],
            "sourceAnalysis": "",
            "redFlags": [],
            "verificationTips": []
        }
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("CREDIBILITY_SCORE:"):
                try:
                    score_str = line.split(":", 1)[1].strip()
                    parsed["credibilityScore"] = int(score_str.split()[0])
                except (ValueError, IndexError):
                    pass
            
            elif line.startswith("RED_FLAGS:"):
                flags_str = line.split(":", 1)[1].strip()
                if flags_str.lower() != "none detected":
                    parsed["redFlags"] = [f.strip() for f in flags_str.split(",") if f.strip()]
            
            elif line.startswith("FACT_CHECK:"):
                claims_str = line.split(":", 1)[1].strip()
                # Split by common separators
                for sep in [";", "|"]:
                    if sep in claims_str:
                        parsed["factCheckSuggestions"] = [c.strip() for c in claims_str.split(sep) if c.strip()]
                        break
                else:
                    # Try numbered list
                    parsed["factCheckSuggestions"] = [claims_str]
            
            elif line.startswith("SOURCE_ANALYSIS:"):
                parsed["sourceAnalysis"] = line.split(":", 1)[1].strip()
            
            elif line.startswith("VERIFICATION_TIPS:"):
                tips_str = line.split(":", 1)[1].strip()
                for sep in [";", "|"]:
                    if sep in tips_str:
                        parsed["verificationTips"] = [t.strip() for t in tips_str.split(sep) if t.strip()]
                        break
                else:
                    parsed["verificationTips"] = [tips_str]
        
        # Handle multi-line sections
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith(("CREDIBILITY_SCORE:", "RED_FLAGS:", "FACT_CHECK:", "SOURCE_ANALYSIS:", "VERIFICATION_TIPS:")):
                current_section = line.split(":", 1)[0]
            elif current_section == "FACT_CHECK" and line and not line.startswith(tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")):
                if line not in parsed["factCheckSuggestions"]:
                    parsed["factCheckSuggestions"].append(line)
            elif current_section == "VERIFICATION_TIPS" and line and not line.startswith(tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")):
                if line not in parsed["verificationTips"]:
                    parsed["verificationTips"].append(line)
        
        return parsed


# Global singleton instance
gemini_service = GeminiInsightService()
