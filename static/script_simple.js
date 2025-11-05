document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const uploadArea = document.getElementById('upload-area');
    const pdfInput = document.getElementById('pdf-input');
    const processingSection = document.getElementById('processing-section');
    const processBtn = document.getElementById('process-btn');
    const loadingSpinner = document.querySelector('.loading-spinner');
    const loadingText = document.getElementById('loading-text');
    const resultsSection = document.getElementById('results-section');
    const resultsTbody = document.getElementById('results-tbody');
    const statusMessages = document.getElementById('status-messages');
    const exportXlsxBtn = document.getElementById('export-xlsx-btn');
    const exportCsvBtn = document.getElementById('export-csv-btn');

    // Global variables
    let currentData = null;
    let csvData = '';

    // File upload handling
    uploadArea.addEventListener('click', () => pdfInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    pdfInput.addEventListener('change', handleFileSelect);

    // Process button
    processBtn.addEventListener('click', processWithAI);

    // Export buttons
    exportXlsxBtn.addEventListener('click', exportXlsx);
    exportCsvBtn.addEventListener('click', exportCsv);

    function handleDragOver(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    }

    function handleDragLeave(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    }

    function handleDrop(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type === 'application/pdf') {
            handleFile(files[0]);
        } else {
            showError('Please drop a valid PDF file');
        }
    }

    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    }

    function handleFile(file) {
        if (file.type !== 'application/pdf') {
            showError('Please select a PDF file');
            return;
        }

        if (file.size > 50 * 1024 * 1024) {
            showError('File size must be less than 50MB');
            return;
        }

        uploadPDF(file);
    }

    function uploadPDF(file) {
        showLoading('Extracting data from PDF...');

        const formData = new FormData();
        formData.append('pdf', file);

        fetch('/extract', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            
            if (data.success) {
                currentData = data.data;
                showSuccess(`Successfully extracted ${Object.keys(data.data).length} data sections`);
                processingSection.style.display = 'block';
            } else {
                showError(data.error || 'Failed to extract data');
            }
        })
        .catch(error => {
            hideLoading();
            showError('Error uploading file: ' + error.message);
        });
    }

    function processWithAI() {
        if (!currentData) {
            showError('No data to process');
            return;
        }

        showLoading('Processing with AI to add context...');

        fetch('/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(currentData)
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            
            if (data.success) {
                displayResults(data.result);
                csvData = data.csv_data;
                showSuccess('Processing completed successfully!');
            } else {
                showError(data.error || 'Processing failed');
            }
        })
        .catch(error => {
            hideLoading();
            showError('Error processing data: ' + error.message);
        });
    }

    function displayResults(result) {
        const entities = result.enhanced_data_with_comprehensive_context || [];
        
        resultsTbody.innerHTML = '';
        
        entities.forEach(entity => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><strong>${entity.field || 'Unknown'}</strong></td>
                <td>${entity.value || ''}</td>
                <td class="context-cell">${entity.context || '<em>No context available</em>'}</td>
            `;
            resultsTbody.appendChild(row);
        });

        resultsSection.style.display = 'block';
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    function exportXlsx() {
        if (!csvData) {
            showError('No data to export. Please process a document first.');
            return;
        }

        showLoading('Generating XLSX file...');

        fetch('/export_xlsx', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ csv_data: csvData })
        })
        .then(response => {
            hideLoading();
            
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Failed to generate XLSX');
                });
            }
            
            return response.blob();
        })
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'extracted_data_with_context.xlsx';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);
            
            showSuccess('XLSX file downloaded successfully!');
        })
        .catch(error => {
            hideLoading();
            showError('Export failed: ' + error.message);
        });
    }

    function exportCsv() {
        if (!csvData) {
            showError('No data to export. Please process a document first.');
            return;
        }

        fetch('/export_csv', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ csv_data: csvData })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Failed to generate CSV');
                });
            }
            
            return response.blob();
        })
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'extracted_data_with_context.csv';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);
            
            showSuccess('CSV file downloaded successfully!');
        })
        .catch(error => {
            showError('Export failed: ' + error.message);
        });
    }

    function showLoading(message) {
        loadingText.textContent = message;
        loadingSpinner.style.display = 'block';
    }

    function hideLoading() {
        loadingSpinner.style.display = 'none';
    }

    function showSuccess(message) {
        showMessage(message, 'success');
    }

    function showError(message) {
        showMessage(message, 'danger');
    }

    function showMessage(message, type) {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        statusMessages.appendChild(alertDiv);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
});