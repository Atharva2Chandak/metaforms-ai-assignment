# api.py
import json
from typing import Any, Dict, List, Type, Union
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field, ValidationError
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import OutputFixingParser
from pydantic import BaseModel as LCBaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter, RecursiveJsonSplitter
import PyPDF2
import marko

app = FastAPI()

class TokenUsage(BaseModel):
    schema_tokens: int
    prompt_tokens: int
    total_input_tokens: int
    output_tokens: int
    total_tokens: int

class SchemaAnalysis(BaseModel):
    max_nesting: int
    total_fields: int
    complexity_score: int
    processing_strategy: str

class ProcessingResult(BaseModel):
    schema_analysis: SchemaAnalysis
    extracted_data: Dict[str, Any]
    validation_passed: bool
    low_confidence_fields: List[str]
    raw_text: str
    correction_attempts: int = 0
    token_usage: TokenUsage


def estimate_tokens(text: str) -> int:
    """Estimate token count using approximation: ~4 characters per token"""
    return max(1, len(text) // 4)

def read_file(file: UploadFile) -> str:
    content = file.file.read()
    try:
        if file.filename.endswith('.pdf'):
            pdf = PyPDF2.PdfReader(file.file)
            return "\n".join(page.extract_text() for page in pdf.pages)
        elif file.filename.endswith('.md'):
            return marko.convert(content.decode())
        else:
            return content.decode()
    except Exception as e:
        raise HTTPException(500, f"File processing error: {str(e)}")

def create_dynamic_model(schema: Dict, model_name: str = "DynamicModel") -> Type[LCBaseModel]:
    def json_to_pydantic_field(field_schema):
        field_type = str
        description = field_schema.get("description", "")
        
        if field_schema.get("type") == "integer":
            field_type = int
        elif field_schema.get("type") == "number":
            field_type = float
        elif field_schema.get("type") == "boolean":
            field_type = bool
        elif field_schema.get("type") == "array":
            items = field_schema.get("items", {})
            if items.get("type") == "object":
                nested_model = json_to_pydantic(f"Nested{field_schema.get('title', 'Item')}", items)
                field_type = List[nested_model]
            elif items.get("type") == "string":
                field_type = List[str]
            elif items.get("type") == "integer":
                field_type = List[int]
            else:
                field_type = List[Any]
        elif field_schema.get("type") == "object":
            field_type = json_to_pydantic(f"Nested{field_schema.get('title', 'Object')}", field_schema)
        
        return (field_type, Field(description=description))
    
    def json_to_pydantic(name: str, schema: Dict) -> Type[LCBaseModel]:
        fields = {}
        properties = schema.get("properties", {})
        
        for field_name, field_schema in properties.items():
            fields[field_name] = json_to_pydantic_field(field_schema)
        
        class DynamicModel(LCBaseModel):
            pass
        
        for field_name, (field_type, field_info) in fields.items():
            setattr(DynamicModel, field_name, field_info)
            DynamicModel.__annotations__[field_name] = field_type
        
        DynamicModel.__name__ = name
        return DynamicModel
    
    return json_to_pydantic(model_name, schema)

def analyze_schema_complexity(schema: Dict) -> SchemaAnalysis:
    def count_levels(obj: Dict, current_level: int = 0) -> int:
        max_level = current_level
        if "properties" in obj:
            for prop in obj["properties"].values():
                level = count_levels(prop, current_level + 1)
                max_level = max(max_level, level)
        return max_level

    def count_fields(obj: Dict) -> int:
        count = len(obj.get("properties", {}))
        return count + sum(count_fields(p) for p in obj.get("properties", {}).values())

    total_fields = count_fields(schema)
    max_nesting = count_levels(schema)
    complexity_score = max_nesting * total_fields
    
    if complexity_score < 10:
        strategy = "single_pass"
    elif complexity_score < 100:
        strategy = "hierarchical"
    else:
        strategy = "progressive_refinement"
    
    return SchemaAnalysis(
        max_nesting=max_nesting,
        total_fields=total_fields,
        complexity_score=complexity_score,
        processing_strategy=strategy
    )

def smart_chunk_text(text: str, schema: Dict, max_chunk_size: int = 8000) -> List[str]:
    """
    Smart chunking that handles both regular text and JSON schemas appropriately
    """
    # Check if the text looks like JSON
    try:
        json_data = json.loads(text)
        # If it's valid JSON, use JSON splitter
        json_splitter = RecursiveJsonSplitter(max_chunk_size=max_chunk_size)
        json_chunks = json_splitter.split_text(json_data=json_data)
        return json_chunks
    except (json.JSONDecodeError, TypeError):
        # If it's not JSON, use regular text chunking
        if len(text) <= max_chunk_size:
            return [text]
        
        # Use RecursiveCharacterTextSplitter for better text chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return text_splitter.split_text(text)

def chunk_schema_for_processing(schema: Dict, max_size: int = 4000) -> List[Dict]:
    """
    Chunk large schemas into smaller, processable pieces while maintaining structure
    """
    schema_str = json.dumps(schema)
    if len(schema_str) <= max_size:
        return [schema]
    
    # Use JSON splitter for schema chunking
    json_splitter = RecursiveJsonSplitter(max_chunk_size=max_size)
    try:
        schema_chunks = json_splitter.split_json(json_data=schema)
        return schema_chunks
    except Exception:
        # Fallback: split by top-level properties
        properties = schema.get("properties", {})
        chunks = []
        current_chunk = {"type": "object", "properties": {}}
        current_size = 0
        
        for prop_name, prop_schema in properties.items():
            prop_str = json.dumps({prop_name: prop_schema})
            if current_size + len(prop_str) > max_size and current_chunk["properties"]:
                chunks.append(current_chunk)
                current_chunk = {"type": "object", "properties": {}}
                current_size = 0
            
            current_chunk["properties"][prop_name] = prop_schema
            current_size += len(prop_str)
        
        if current_chunk["properties"]:
            chunks.append(current_chunk)
        
        return chunks if chunks else [schema]

@app.post("/process", response_model=ProcessingResult)
async def process_file(
    file: UploadFile = File(...),
    schema_file: UploadFile = File(...),
    api_key: str = "AIzaSyA7j97VTnc97NBhZtfai6MQPgXzUHUCTRc"
):
    try:
        # Read and validate inputs
        text_content = read_file(file)
        schema = json.loads(await schema_file.read())
        
        # Schema analysis
        analysis = analyze_schema_complexity(schema)

        # Token estimation
        schema_text = json.dumps(schema, indent=2)
        schema_tokens = estimate_tokens(schema_text)
        prompt_tokens = estimate_tokens(text_content)
        
        # Create dynamic Pydantic model
        model_class = create_dynamic_model(schema)
        
        # Initialize Gemini model
        if analysis.complexity_score > 10:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=api_key,
                temperature=0.1
            )
        else:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-lite",
                google_api_key=api_key,
                temperature=0.1
            )
        
        # Create base parser
        base_parser = PydanticOutputParser(pydantic_object=model_class)
        
        # Create output fixing parser
        fixing_parser = OutputFixingParser.from_llm(
            parser=base_parser,
            llm=llm,
            max_retries=3,
        )
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_template(
            """You are an expert data extraction specialist. Extract structured information from the following document according to the specified schema.
            Response should be in JSON format. no other preamble or descriptions.

            Document Content:
            {text}
            
            Schema to follow:
            {schema_desc}
            
            Extract the information following this exact format:
            {format_instructions}
            
            Important: 
            - Extract information that is explicitly mentioned in the document no hallucinations
            - Try to extract MAXIMUM information as possible
            - If a field is not found, return null value if it is a required field else skip it
            - MOST IMPORTANTLY: Ensure all extracted data matches the schema requirements
            
            Extracted Information:"""
        )
        
        # Build processing chain with fixing parser
        chain = prompt.partial(format_instructions=base_parser.get_format_instructions()) | llm | fixing_parser

        # Smart chunking based on content type and schema complexity
        if analysis.complexity_score > 50:
            # For very complex schemas, chunk the schema itself
            schema_chunks = chunk_schema_for_processing(schema, max_size=4000)
            text_chunks = smart_chunk_text(text_content, schema, max_chunk_size=6000)
        else:
            schema_chunks = [schema]
            text_chunks = smart_chunk_text(text_content, schema, max_chunk_size=8000)
        
        results = []
        correction_attempts = 0
        total_output_tokens = 0
        # Process each combination of text and schema chunks
        for schema_chunk in schema_chunks:
            for text_chunk in text_chunks:
                try:
                    result = chain.invoke({
                        "text": text_chunk,
                        "schema_desc": json.dumps(schema_chunk, indent=2)
                    })
                    if hasattr(result, 'dict'):
                        results.append(result.dict())
                    else:
                        results.append(result)
                    # Estimate output tokens
                    total_output_tokens += estimate_tokens(json.dumps(result.dict() if hasattr(result, 'dict') else result))
                except Exception as e:
                    correction_attempts += 1
                    # Fallback: try with simpler prompt
                    simple_prompt = ChatPromptTemplate.from_template(
                        "Extract key information from this text as JSON matching the schema: {text}\nSchema: {schema_desc}"
                    )
                    fallback_chain = simple_prompt | llm | fixing_parser
                    result = fallback_chain.invoke({
                        "text": text_chunk,
                        "schema_desc": json.dumps(schema_chunk, indent=2)
                    })
                    if hasattr(result, 'dict'):
                        results.append(result.dict())
                    else:
                        results.append(result)
                    
                    total_output_tokens += estimate_tokens(json.dumps(result.dict() if hasattr(result, 'dict') else result))
        
        # Merge results intelligently
        if len(results) == 1:
            final_result = results[0]
        else:
            final_result = merge_chunk_results(results)
        
        # Calculate token usage
        total_input_tokens = schema_tokens + prompt_tokens
        token_usage = TokenUsage(
            schema_tokens=schema_tokens,
            prompt_tokens=prompt_tokens,
            total_input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            total_tokens=total_input_tokens + total_output_tokens
        )
        # Identify low confidence fields
        low_confidence = identify_low_confidence_fields(final_result)
        print(final_result, low_confidence, correction_attempts, analysis)
        return ProcessingResult(
            schema_analysis=analysis,
            extracted_data=final_result,
            validation_passed=True,
            low_confidence_fields=low_confidence,
            raw_text=text_content[:10000] + "..." if len(text_content) > 10000 else text_content,
            token_usage=token_usage,
            correction_attempts=correction_attempts
        )
        
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON schema format")
    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")

def merge_chunk_results(results: List[Dict]) -> Dict:
    """Merge results from multiple chunks intelligently"""
    if not results:
        return {}
    
    merged = results[0].copy()
    
    for result in results[1:]:
        for key, value in result.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(value, list) and isinstance(merged[key], list):
                # Merge lists, avoiding duplicates
                merged[key].extend([item for item in value if item not in merged[key]])
            elif isinstance(value, dict) and isinstance(merged[key], dict):
                # Merge dictionaries recursively
                merged[key].update(value)
            elif value and not merged[key]:
                # Replace empty/null values with non-empty ones
                merged[key] = value
    
    return merged

def identify_low_confidence_fields(data: Dict, threshold: int = 3) -> List[str]:
    """Identify fields that might have low confidence based on simple heuristics"""
    low_confidence = []
    
    def check_field(obj, path=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                if isinstance(value, str) and len(value) < threshold:
                    low_confidence.append(current_path)
                elif isinstance(value, list) and len(value) == 0:
                    low_confidence.append(current_path)
                elif isinstance(value, dict):
                    check_field(value, current_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                check_field(item, f"{path}[{i}]")
    
    check_field(data)
    return low_confidence

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
