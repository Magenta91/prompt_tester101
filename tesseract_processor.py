import pytesseract
import fitz  # PyMuPDF
from PIL import Image
import io
import time
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TesseractProcessor:
    def __init__(self):
        """Initialize Tesseract processor"""
        # You may need to set the tesseract path on Windows
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        pass

    def extract_text_from_pdf_bytes(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """
        Extract structured data from PDF bytes using Tesseract OCR as fallback.
        
        Args:
            pdf_bytes (bytes): PDF file as bytes
            
        Returns:
            Dict[str, Any]: Structured JSON with document_text, tables, and key_values
        """
        start_time = time.time()
        
        try:
            print("Using Tesseract OCR for PDF processing (fallback mode)")
            
            # Convert PDF to images using PyMuPDF
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            all_document_text = []
            tables = []
            key_values = []
            footnotes = []
            
            # Process each page
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                
                # Convert page to image
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR quality
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(img_data))
                
                # Extract text using Tesseract
                page_text = pytesseract.image_to_string(image, config='--psm 6')
                
                # Split into lines and clean
                lines = [line.strip() for line in page_text.split('\n') if line.strip()]
                all_document_text.extend(lines)
                
                # Try to extract table-like data using Tesseract's table detection
                try:
                    # Use TSV output to detect table structure
                    tsv_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                    table_data = self._extract_table_from_tsv(tsv_data, page_num + 1)
                    if table_data:
                        tables.append(table_data)
                except Exception as e:
                    print(f"Table extraction failed for page {page_num + 1}: {e}")
                
                # Simple key-value extraction based on patterns
                kv_pairs = self._extract_key_values_from_text(lines, page_num + 1)
                key_values.extend(kv_pairs)
            
            pdf_document.close()
            
            # Enhanced footnote processing
            footnote_analysis = self._enhance_footnote_detection(all_document_text)
            
            processing_time = f"{time.time() - start_time:.1f}s"
            print(f"Tesseract OCR processing completed in {processing_time}")
            print(f"Found {len(footnote_analysis['footnotes'])} footnotes")
            
            return {
                "document_text": all_document_text,
                "tables": tables,
                "key_values": key_values,
                "footnotes": footnote_analysis['footnotes'],
                "footnote_markers": footnote_analysis['footnote_markers'],
                "enhanced_text": footnote_analysis['enhanced_text']
            }
            
        except Exception as e:
            print(f"Tesseract OCR extraction failed: {e}")
            raise Exception(f"Failed to extract text using Tesseract OCR: {str(e)}")

    def _extract_table_from_tsv(self, tsv_data: Dict, page_num: int) -> Optional[Dict[str, Any]]:
        """Extract table structure from Tesseract TSV output"""
        try:
            # Group text by block and line
            blocks = {}
            for i, block_num in enumerate(tsv_data['block_num']):
                if tsv_data['text'][i].strip():
                    if block_num not in blocks:
                        blocks[block_num] = {}
                    
                    line_num = tsv_data['line_num'][i]
                    if line_num not in blocks[block_num]:
                        blocks[block_num][line_num] = []
                    
                    blocks[block_num][line_num].append({
                        'text': tsv_data['text'][i].strip(),
                        'left': tsv_data['left'][i],
                        'top': tsv_data['top'][i],
                        'width': tsv_data['width'][i],
                        'height': tsv_data['height'][i]
                    })
            
            # Convert to table format if we have structured data
            if len(blocks) > 1:  # Multiple blocks might indicate table structure
                rows = []
                for block_num in sorted(blocks.keys()):
                    for line_num in sorted(blocks[block_num].keys()):
                        # Sort words by left position to maintain column order
                        words = sorted(blocks[block_num][line_num], key=lambda x: x['left'])
                        row = [word['text'] for word in words]
                        if len(row) > 1:  # Only include rows with multiple columns
                            rows.append(row)
                
                if rows:
                    return {
                        "page": page_num,
                        "rows": rows
                    }
            
            return None
            
        except Exception as e:
            print(f"Table extraction from TSV failed: {e}")
            return None

    def _extract_key_values_from_text(self, lines: List[str], page_num: int) -> List[Dict[str, Any]]:
        """Extract key-value pairs from text lines using pattern matching"""
        import re
        
        key_values = []
        
        # Common key-value patterns
        patterns = [
            r'^([A-Za-z\s]+):\s*(.+)$',  # "Key: Value"
            r'^([A-Za-z\s]+)\s+([0-9,.$%]+)$',  # "Key 123.45"
            r'^([A-Za-z\s]+)\s*[-–]\s*(.+)$',  # "Key - Value"
        ]
        
        for line in lines:
            line = line.strip()
            if len(line) < 5:  # Skip very short lines
                continue
                
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    key = match.group(1).strip()
                    value = match.group(2).strip()
                    
                    # Filter out common non-key-value patterns
                    if (len(key) > 2 and len(value) > 0 and 
                        not key.lower().startswith(('page', 'figure', 'table', 'section'))):
                        key_values.append({
                            "key": key,
                            "value": value,
                            "page": page_num
                        })
                    break
        
        return key_values

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


def extract_text_from_pdf_bytes_tesseract(pdf_bytes: bytes) -> str:
    """
    Main function to extract raw text from PDF bytes using Tesseract OCR.
    Returns document_text joined for backward compatibility.
    
    Args:
        pdf_bytes (bytes): PDF file as bytes
        
    Returns:
        str: Raw extracted text from the PDF
    """
    processor = TesseractProcessor()
    result = processor.extract_text_from_pdf_bytes(pdf_bytes)
    return '\n'.join(result.get('document_text', []))


def extract_structured_data_from_pdf_bytes_tesseract(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Main function to extract structured data from PDF bytes using Tesseract OCR.
    
    Args:
        pdf_bytes (bytes): PDF file as bytes
        
    Returns:
        Dict[str, Any]: Structured JSON with document_text, tables, and key_values
    """
    processor = TesseractProcessor()
    return processor.extract_text_from_pdf_bytes(pdf_bytes)