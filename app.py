# app.py
import streamlit as st
import requests
import json
import time
from typing import Optional

def main():
    st.set_page_config(page_title="Doc2JSON Converter", layout="wide")
    
    # Sidebar for API configuration
    with st.sidebar:
        st.header("API Configuration")
        api_key = st.text_input(
            "Gemini API Key", 
            value="",
            type="password"
        )
        api_url = st.text_input("Backend URL", "http://localhost:8000/process")
        st.markdown("---")
        st.caption("Upload your documents and schemas to extract structured data")
    
    # Main interface
    st.title("📄 Document to Structured Data Converter")
    st.markdown("Convert unstructured documents to structured JSON using AI-powered schema matching")
    
    # File upload section
    col1, col2 = st.columns([2, 3])
    with col1:
        st.subheader("Upload Files")
        file = st.file_uploader("Document")
        schema_file = st.file_uploader("JSON Schema", type=["json"])
    
    with col2:
        st.subheader("Example Schemas")
        tab1, tab2, tab3 = st.tabs(["Resume", "Research Paper", "GitHub Action"])
        
        with tab1:
            st.code("""{
  "type": "object",
  "properties": {
    "basics": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "email": {"type": "string"},
        "phone": {"type": "string"}
      }
    },
    "work": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "position": {"type": "string"},
          "startDate": {"type": "string"},
          "endDate": {"type": "string"}
        }
      }
    }
  }
}""", language="json")
        
        with tab2:
            st.code("""{
  "type": "object",
  "properties": {
    "title": {"type": "string"},
    "authors": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "given-names": {"type": "string"},
          "family-names": {"type": "string"}
        }
      }
    }
  }
}""", language="json")
        
        with tab3:
            st.code("""{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "description": {"type": "string"},
    "inputs": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "properties": {
          "description": {"type": "string"},
          "required": {"type": "boolean"}
        }
      }
    }
  }
}""", language="json")

    # Always show intermediate processing sections
    show_intermediate_sections(file, schema_file)

    # Processing controls
    if st.button("✨ Process Document", use_container_width=True, type="primary"):
        if not file or not schema_file:
            st.warning("⚠️ Please upload both document and schema files")
            return
            
        if not api_key.strip():
            st.error("⚠️ Please provide a Gemini API Key in the sidebar")
            return

        with st.status("Processing document...", expanded=True) as status:
            try:
                st.write("📤 Uploading files to API...")
                
                file.seek(0)
                schema_file.seek(0)
                
                files = {
                    "file": (file.name, file.getvalue(), file.type),
                    "schema_file": (schema_file.name, schema_file.getvalue(), "application/json")
                }
                params = {"api_key": api_key}
                
                progress_bar = st.progress(0)
                
                start_time = time.time()
                st.write("🔄 Processing...")
                progress_bar.progress(30)
                
                response = requests.post(api_url, files=files, params=params, timeout=120)
                progress_bar.progress(70)
                
                if response.status_code != 200:
                    st.error(f"❌ API Error ({response.status_code}): {response.text}")
                    return

                result = response.json()
                progress_bar.progress(100)
                
                st.write("📊 Analyzing results...")
                display_results(result)
                
                status.update(label="✅ Processing complete!", state="complete")
                
                processing_time = time.time() - start_time
                st.success(f"🎉 Processed successfully in {processing_time:.2f}s")
                
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Please try with a smaller document.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Cannot connect to API. Make sure the backend is running.")
            except Exception as e:
                st.error(f"💥 Processing failed: {str(e)}")

def show_intermediate_sections(file, schema_file):
    """Show intermediate processing information regardless of API call status"""
    
    st.divider()
    st.header("📋 Processing Preview")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📄 Document Information")
        if file:
            st.write(f"**Filename:** {file.name}")
            st.write(f"**File Type:** {file.type}")
            st.write(f"**File Size:** {file.size:,} bytes (Max size: 10MB)")
            
            # Estimate tokens for document
            if file.type == "text/plain" or file.name.endswith('.md'):
                file_content = file.getvalue().decode()
                estimated_tokens = len(file_content) // 4
                st.write(f"**Estimated Tokens:** ~{estimated_tokens:,}")
                st.text_area("Text Preview", file_content[:500] + "..." if len(file_content) > 500 else file_content, height=150, disabled=True)
        else:
            st.info("No document uploaded yet")
    
    with col2:
        st.subheader("🔧 Schema Information")
        if schema_file:
            try:
                schema_content = json.loads(schema_file.getvalue().decode())
                st.write(f"**Schema Type:** {schema_content.get('type', 'Unknown')}")
                
                properties = schema_content.get('properties', {})
                st.write(f"**Top-level Properties:** {len(properties)}")
                
                # Estimate tokens for schema
                schema_text = json.dumps(schema_content, indent=2)
                estimated_schema_tokens = len(schema_text) // 4
                st.write(f"**Estimated Schema Tokens:** ~{estimated_schema_tokens:,}")
                
                st.json(schema_content, expanded=False)
                
            except json.JSONDecodeError:
                st.error("Invalid JSON schema format")
        else:
            st.info("No schema uploaded yet")

def display_results(result: dict):
    """Display processing results with enhanced formatting"""
    
    st.divider()
    
    # Token Usage Section
    st.header("🔢 Token Usage Analysis")
    token_usage = result.get("token_usage", {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Schema Tokens", f"{token_usage.get('schema_tokens', 0):,}")
    with col2:
        st.metric("Prompt Tokens", f"{token_usage.get('prompt_tokens', 0):,}")
    with col3:
        st.metric("Output Tokens", f"{token_usage.get('output_tokens', 0):,}")
    with col4:
        st.metric("Total Tokens", f"{token_usage.get('total_tokens', 0):,}")
    
    # Cost estimation (approximate)
    total_tokens = token_usage.get('total_tokens', 0)
    estimated_cost = total_tokens * 0.00001  # Rough estimate
    
    st.divider()

    
    # Schema Analysis Section
    st.header("🔍 Schema Analysis")
    analysis = result["schema_analysis"]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Max Nesting", analysis["max_nesting"])
    with col2:
        st.metric("Total Fields", analysis["total_fields"])
    with col3:
        st.metric("Complexity Score", analysis["complexity_score"])
    with col4:
        st.metric("Strategy", analysis["processing_strategy"])
    
    complexity = analysis["complexity_score"]
    if complexity < 10:
        st.success("🟢 Low complexity schema")
    elif complexity < 100:
        st.warning("🟡 Medium complexity schema")
    else:
        st.error("🔴 High complexity schema")
    
    st.divider()
    
    # Processing Details Section
    st.header("⚙️ Processing Details")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Validation Status")
        if result["validation_passed"]:
            st.success("✅ All fields passed validation")
        else:
            st.error("❌ Validation failed for some fields")
    
    with col2:
        st.subheader("Correction Attempts")
        attempts = result.get("correction_attempts", 0)
        
        st.warning(f"🔄 Required {attempts} correction attempts")
    
    st.divider()
    
    # Low Confidence Fields Section
    if result.get("low_confidence_fields"):
        st.header("⚠️ Low Confidence Fields")
        st.warning("The following fields may need manual review:")
        for field in result["low_confidence_fields"]:
            st.write(f"• **{field}**")
        st.divider()
    
    # Raw Text Preview Section
    st.header("📄 Raw Text Preview")
    with st.container():
        st.text_area(
            "Extracted Text",
            result["raw_text"],
            height=200,
            disabled=True
        )
    
    st.divider()
    
    # Structured Output Section
    st.header("📦 Structured Output")
    st.json(result["extracted_data"])
    
    # Download button for extracted data
    json_str = json.dumps(result["extracted_data"], indent=2)
    st.download_button(
        label="📥 Download JSON",
        data=json_str,
        file_name="extracted_data.json",
        mime="application/json"
    )

if __name__ == "__main__":
    main()
