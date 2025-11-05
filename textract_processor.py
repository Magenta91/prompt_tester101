import boto3
import time
import uuid
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class TextractProcessor:
    def __init__(self):
        """Initialize AWS Textract and S3 clients with credentials from environment"""
        self.textract_client = boto3.client('textract')
        self.s3_client = boto3.client('s3')
        self.bucket_name = 'textract-bucket-lk'

    def extract_text_from_pdf_bytes(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """
        Extract structured data from PDF bytes using Amazon Textract with S3 storage.
        
        Args:
            pdf_bytes (bytes): PDF file as bytes
            
        Returns:
            Dict[str, Any]: Structured JSON with document_text, tables, and key_values
        """
        start_time = time.time()
        
        try:
            print("Using Amazon Textract with S3 storage for PDF processing")
            
            # Upload PDF to S3
            file_key = f"textract-input/{uuid.uuid4()}.pdf"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=file_key,
                Body=pdf_bytes,
                ContentType='application/pdf'
            )
            
            # Start document analysis
            response = self.textract_client.start_document_analysis(
                DocumentLocation={
                    'S3Object': {
                        'Bucket': self.bucket_name,
                        'Name': file_key
                    }
                },
                FeatureTypes=['TABLES', 'FORMS']
            )
            
            job_id = response['JobId']
            print(f"Started Textract job: {job_id}")
            
            # Wait for job completion
            while True:
                result = self.textract_client.get_document_analysis(JobId=job_id)
                status = result['JobStatus']
                print(f"Job status: {status}")
                
                if status in ['SUCCEEDED', 'FAILED']:
                    break
                
                time.sleep(5)
            
            if status == 'FAILED':
                raise Exception("Textract job failed")
            
            # Fetch full results (handle pagination)
            pages = []
            next_token = None
            
            while True:
                if next_token:
                    response = self.textract_client.get_document_analysis(JobId=job_id, NextToken=next_token)
                else:
                    response = self.textract_client.get_document_analysis(JobId=job_id)
                
                pages.extend(response['Blocks'])
                
                next_token = response.get('NextToken')
                if not next_token:
                    break
            
            print(f"Total blocks extracted: {len(pages)}")
            
            # Parse the results
            structured_data = self._parse_textract_blocks(pages, start_time)
            
            # Clean up S3 file
            try:
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=file_key)
            except Exception as e:
                print(f"Warning: Could not delete S3 file: {e}")
            
            return structured_data
            
        except Exception as e:
            print(f"Textract extraction failed: {e}")
            raise Exception(f"Failed to extract text using Amazon Textract: {str(e)}")

    def _enhance_footnote_detection(self, document_text):
        """Enhanced footnote detection and processing"""
        import re
        
        footnotes = []
        enhanced_text = []
        footnote_markers = {}
        
        # Common footnote patterns
        footnote_patterns = [
            r'^\(\d+\)',  # (1), (2) at start of line
            r'^\[\d+\]',  # [1], [2] at start of line
            r'^\d+\.',    # 1., 2., 3. at start of line
            r'^\*+\s',    # *, **, *** at start with space
            r'^Note\s*\d*:',  # Note: or Note 1:
            r'^Source:',  # Source:
            r'^See\s',    # See ...
        ]
        
        for i, line in enumerate(document_text):
            line_stripped = line.strip()
            if not line_stripped:
                continue
                
            # Check if this line is a footnote
            is_footnote = False
            footnote_marker = None
            
            for pattern in footnote_patterns:
                match = re.match(pattern, line_stripped, re.IGNORECASE)
                if match:
                    footnote_marker = match.group()
                    # Additional checks for footnote characteristics
                    if (len(line_stripped) > len(footnote_marker) + 5 and  # Has content after marker
                        (any(word in line_stripped.lower() for word in 
                             ['note', 'source', 'see', 'reference', 'pursuant', 'accordance', 
                              'disclaimer', 'based on', 'refers to', 'includes', 'excludes']) or
                         re.search(r'\b(?:page|section|chapter|exhibit|appendix)\s+\d+', line_stripped.lower()))):
                        is_footnote = True
                        break
            
            if is_footnote:
                footnotes.append({
                    'marker': footnote_marker,
                    'content': line_stripped,
                    'line_number': i,
                    'type': 'footnote'
                })
                footnote_markers[footnote_marker] = line_stripped
            else:
                # Check for inline footnote references
                has_refs = bool(re.search(r'[\(\[]\d+[\)\]]|\*+(?=\s|$)', line_stripped))
                enhanced_text.append({
                    'content': line_stripped,
                    'has_footnote_refs': has_refs,
                    'line_number': i
                })
        
        return {
            'enhanced_text': enhanced_text,
            'footnotes': footnotes,
            'footnote_markers': footnote_markers
        }

    def _remove_superscript_numbers(self, text):
        """Remove superscript numbers and common footnote markers from text"""
        import re
        
        # Remove superscript numbers (Unicode superscript characters)
        superscript_pattern = r'[⁰¹²³⁴⁵⁶⁷⁸⁹]+'
        text = re.sub(superscript_pattern, '', text)
        
        # Remove common footnote reference patterns
        footnote_patterns = [
            r'\(\d+\)',    # (1), (2), etc.
            r'\[\d+\]',    # [1], [2], etc.
            r'\*+',        # *, **, ***, etc.
            r'^\d+$',      # Standalone numbers on their own line
        ]
        
        for pattern in footnote_patterns:
            text = re.sub(pattern, '', text)
        
        # Clean up extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()

    def _parse_textract_blocks(self, blocks: List[Dict[str, Any]], start_time: float) -> Dict[str, Any]:
        """Parse Textract blocks into the specified JSON format with enhanced footnote handling"""
        
        block_map = {block['Id']: block for block in blocks}
        
        # Extract document text (line by line)
        raw_document_text = []
        tables = []
        key_values = []
        
        # Process blocks by page to maintain order
        pages_blocks = {}
        for block in blocks:
            page_num = block.get('Page', 1)
            if page_num not in pages_blocks:
                pages_blocks[page_num] = []
            pages_blocks[page_num].append(block)
        
        # Process each page in order
        all_document_text = []
        for page_num in sorted(pages_blocks.keys()):
            page_blocks = pages_blocks[page_num]
            
            # Sort blocks by geometry (top to bottom, left to right)
            line_blocks = [b for b in page_blocks if b['BlockType'] == 'LINE']
            line_blocks.sort(key=lambda x: (
                x.get('Geometry', {}).get('BoundingBox', {}).get('Top', 0),
                x.get('Geometry', {}).get('BoundingBox', {}).get('Left', 0)
            ))
            
            for block in line_blocks:
                text = block.get('Text', '').strip()
                if text:
                    # Remove superscript numbers (common footnote references)
                    cleaned_text = self._remove_superscript_numbers(text)
                    if cleaned_text:
                        all_document_text.append(cleaned_text)
            
            # Process other block types for this page
            for block in page_blocks:
                if block['BlockType'] == 'TABLE':
                    table_data = self._extract_table_structure(block, block_map)
                    if table_data:
                        tables.append(table_data)
                
                elif block['BlockType'] == 'KEY_VALUE_SET':
                    kv_pair = self._extract_key_value_pair(block, block_map)
                    if kv_pair:
                        key_values.append(kv_pair)
        
        # Enhanced footnote processing
        footnote_analysis = self._enhance_footnote_detection(all_document_text)
        
        processing_time = f"{time.time() - start_time:.1f}s"
        print(f"Textract processing completed in {processing_time}")
        print(f"Found {len(footnote_analysis['footnotes'])} footnotes")
        
        return {
            "document_text": all_document_text,
            "tables": tables,
            "key_values": key_values,
            "footnotes": footnote_analysis['footnotes'],
            "footnote_markers": footnote_analysis['footnote_markers'],
            "enhanced_text": footnote_analysis['enhanced_text']
        }

    def _extract_table_structure(self, table_block: Dict[str, Any], block_map: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract table as rows format"""
        if 'Relationships' not in table_block:
            return None
        
        # Get page number
        page_num = table_block.get('Page', 1)
        
        # Find all cells in the table
        cells = []
        for relationship in table_block['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    if child_id in block_map and block_map[child_id]['BlockType'] == 'CELL':
                        cells.append(block_map[child_id])
        
        if not cells:
            return None
        
        # Organize cells by row and column
        table_structure = {}
        max_row = 0
        max_col = 0
        
        for cell in cells:
            row_index = cell.get('RowIndex', 1) - 1  # Convert to 0-based
            col_index = cell.get('ColumnIndex', 1) - 1  # Convert to 0-based
            
            max_row = max(max_row, row_index)
            max_col = max(max_col, col_index)
            
            cell_text = self._get_cell_text(cell, block_map)
            
            if row_index not in table_structure:
                table_structure[row_index] = {}
            table_structure[row_index][col_index] = cell_text
        
        # Convert to list of lists (rows)
        rows = []
        for row_idx in range(max_row + 1):
            row = []
            for col_idx in range(max_col + 1):
                cell_value = table_structure.get(row_idx, {}).get(col_idx, "")
                row.append(cell_value)
            rows.append(row)
        
        return {
            "page": page_num,
            "rows": rows
        }

    def _get_cell_text(self, cell_block: Dict[str, Any], block_map: Dict[str, Any]) -> str:
        """Extract text from a table cell"""
        if 'Relationships' not in cell_block:
            return ""
        
        text_parts = []
        for relationship in cell_block['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    if child_id in block_map:
                        child_block = block_map[child_id]
                        if child_block['BlockType'] == 'WORD':
                            text_parts.append(child_block.get('Text', ''))
        
        return ' '.join(text_parts)

    def _extract_key_value_pair(self, kv_block: Dict[str, Any], block_map: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract key-value pairs"""
        if kv_block.get('EntityTypes') and 'KEY' in kv_block['EntityTypes']:
            page_num = kv_block.get('Page', 1)
            
            key_text = self._get_text_from_block(kv_block, block_map)
            value_text = ""
            
            # Find the corresponding VALUE
            if 'Relationships' in kv_block:
                for relationship in kv_block['Relationships']:
                    if relationship['Type'] == 'VALUE':
                        for value_id in relationship['Ids']:
                            if value_id in block_map:
                                value_block = block_map[value_id]
                                value_text = self._get_text_from_block(value_block, block_map)
                                break
            
            if key_text:
                return {
                    "key": key_text,
                    "value": value_text,
                    "page": page_num
                }
        
        return None

    def _get_text_from_block(self, block: Dict[str, Any], block_map: Dict[str, Any]) -> str:
        """Get text content from a block"""
        if 'Text' in block:
            return block['Text']
        
        if 'Relationships' not in block:
            return ""
        
        text_parts = []
        for relationship in block['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    if child_id in block_map:
                        child_block = block_map[child_id]
                        if child_block['BlockType'] in ['WORD', 'LINE']:
                            text_parts.append(child_block.get('Text', ''))
        
        return ' '.join(text_parts)


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Main function to extract raw text from PDF bytes using Amazon Textract with Tesseract fallback.
    Returns document_text joined for backward compatibility.
    
    Args:
        pdf_bytes (bytes): PDF file as bytes
        
    Returns:
        str: Raw extracted text from the PDF
    """
    try:
        # Try Amazon Textract first
        processor = TextractProcessor()
        result = processor.extract_text_from_pdf_bytes(pdf_bytes)
        return '\n'.join(result.get('document_text', []))
    except Exception as textract_error:
        print(f"Amazon Textract failed: {textract_error}")
        print("Falling back to Tesseract OCR...")
        
        try:
            # Fallback to Tesseract OCR
            from tesseract_processor import extract_text_from_pdf_bytes_tesseract
            return extract_text_from_pdf_bytes_tesseract(pdf_bytes)
        except Exception as tesseract_error:
            print(f"Tesseract OCR also failed: {tesseract_error}")
            return f"Error: Both Textract and Tesseract failed. Textract: {str(textract_error)}, Tesseract: {str(tesseract_error)}"


def extract_structured_data_from_pdf_bytes(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Main function to extract structured data from PDF bytes using Amazon Textract with Tesseract fallback.
    
    Args:
        pdf_bytes (bytes): PDF file as bytes
        
    Returns:
        Dict[str, Any]: Structured JSON with document_text, tables, and key_values
    """
    try:
        # Try Amazon Textract first
        processor = TextractProcessor()
        result = processor.extract_text_from_pdf_bytes(pdf_bytes)
        result["extraction_method"] = "Amazon Textract"
        return result
    except Exception as textract_error:
        print(f"Amazon Textract failed: {textract_error}")
        print("Falling back to Tesseract OCR...")
        
        try:
            # Fallback to Tesseract OCR
            from tesseract_processor import extract_structured_data_from_pdf_bytes_tesseract
            result = extract_structured_data_from_pdf_bytes_tesseract(pdf_bytes)
            result["extraction_method"] = "Tesseract OCR (Fallback)"
            result["textract_error"] = str(textract_error)
            return result
        except Exception as tesseract_error:
            print(f"Tesseract OCR also failed: {tesseract_error}")
            # Return a basic structure with error information
            return {
                "document_text": [f"Error: Both Textract and Tesseract failed. Textract: {str(textract_error)}, Tesseract: {str(tesseract_error)}"],
                "tables": [],
                "key_values": [],
                "footnotes": [],
                "footnote_markers": {},
                "enhanced_text": [],
                "extraction_method": "Failed",
                "textract_error": str(textract_error),
                "tesseract_error": str(tesseract_error)
            }


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Main function to extract raw text from PDF file using Amazon Textract.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Raw extracted text from the PDF
    """
    with open(pdf_path, 'rb') as file:
        pdf_bytes = file.read()
    return extract_text_from_pdf_bytes(pdf_bytes)