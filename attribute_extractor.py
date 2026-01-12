
import os
import json
from typing import Dict, List, Optional, Any
from openai import OpenAI
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class AttributeExtractor:
    """
    Extracts product attributes using LLM from product title, description, and image.
    """
    
    def __init__(self, 
                 provider: str = 'openai',
                 model: Optional[str] = None,
                 api_key: Optional[str] = None):
        """
        Initialize attribute extractor with OpenAI.
        
        Args:
            provider: LLM provider (only 'openai' supported)
            model: Model name (defaults to 'gpt-4o')
            api_key: API key (if not in environment variables)
        """
        self.provider = provider.lower()
        
        if self.provider == 'openai':
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not found. Set it in environment or pass as parameter.")
            self.client = OpenAI(api_key=self.api_key)
            self.model = model or 'gpt-4.1-mini'
        else:
            raise ValueError(f"Unsupported provider: {provider}. Only 'openai' is supported.")
        
        logger.info(f"Attribute extractor initialized with {self.provider} ({self.model})")
    
    def _create_extraction_prompt(self, title: str, description: str, image_text: str = "") -> str:
        """
        Create prompt for attribute extraction.
        
        Args:
            title: Product title
            description: Product description
            image_text: Text extracted from image
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Extract all possible product attributes from the following product information.

Product Title: {title}

Product Description: {description}
"""
        
        if image_text:
            prompt += f"""
Text extracted from product image: {image_text}
"""
        
        prompt += """
        You are helpful assistant to extract the information from the text provided to you.
Please extract and return a comprehensive list of product attributes in JSON format. Include:
- Basic attributes (name, brand, category, type)
- Physical attributes (dimensions, weight, color, material, size)
- Functional attributes (features, specifications, capabilities)
- Technical attributes (model number, SKU, compatibility, requirements)
- Aesthetic attributes (style, design, finish)
- Any other relevant attributes you can identify

Return the response as a valid JSON object with this structure:
{
  "basic_attributes": {
    "name": "...",
    "brand": "...",
    "category": "...",
    "product_type": "..."
  },
  "physical_attributes": {
    "dimensions": "...",
    "weight": "...",
    "color": "...",
    "material": "...",
    "size": "..."
  },
  "functional_attributes": {
    "features": ["..."],
    "specifications": {...},
    "capabilities": ["..."]
  },
  "technical_attributes": {
    "model_number": "...",
    "sku": "...",
    "compatibility": "...",
    "requirements": "..."
  },
  "aesthetic_attributes": {
    "style": "...",
    "design": "...",
    "finish": "..."
  },
  "other_attributes": {
    "additional_info": {...}
  }
}

Only return the JSON object, no additional text or explanation.
"""
        return prompt
    
    def extract_attributes(self, 
                          title: str, 
                          description: str,
                          image_text: str = "",
                          image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract attributes from product information.
        
        Args:
            title: Product title
            description: Product description
            image_text: Text extracted from image (optional)
            image_path: Path to product image for vision analysis (optional)
            
        Returns:
            Dictionary containing extracted attributes
        """
        logger.info("Extracting product attributes...")
        
        try:
            return self._extract_with_openai(title, description, image_text, image_path)
        except Exception as e:
            logger.error(f"Error extracting attributes: {str(e)}")
            raise
    
    def _extract_with_openai(self, 
                            title: str, 
                            description: str,
                            image_text: str,
                            image_path: Optional[str]) -> Dict[str, Any]:
        """Extract attributes using OpenAI API."""
        prompt = self._create_extraction_prompt(title, description, image_text)
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert product attribute extraction system. Extract all possible attributes from product information and return only valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # If image path provided, add vision capability
        if image_path and os.path.exists(image_path):
            import base64
            with open(image_path, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            messages[1]["content"] = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                }
            ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        # Parse JSON response
        result = json.loads(response.choices[0].message.content)
        logger.info("Attributes extracted successfully")
        return result
    
    def extract_attributes_batch(self, 
                                products: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Extract attributes for multiple products.
        
        Args:
            products: List of product dictionaries with 'title', 'description', 
                     optional 'image_text' and 'image_path'
            
        Returns:
            List of extracted attribute dictionaries
        """
        results = []
        for i, product in enumerate(products):
            logger.info(f"Processing product {i+1}/{len(products)}")
            attributes = self.extract_attributes(
                title=product.get('title', ''),
                description=product.get('description', ''),
                image_text=product.get('image_text', ''),
                image_path=product.get('image_path')
            )
            results.append(attributes)
        
        return results


if __name__ == "__main__":
    extractor = AttributeExtractor(provider='openai')
    
    attributes = extractor.extract_attributes(
        title="Apple iPhone 15 Pro Max 256GB Titanium Blue",
        description="Latest iPhone with A17 Pro chip, 48MP camera, and titanium design.",
        image_text="iPhone 15 Pro Max, 256GB, Titanium"
    )
    
    print(json.dumps(attributes, indent=2))
