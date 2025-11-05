document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const uploadArea = document.getElementById('upload-area');
    const pdfInput = document.getElementById('pdf-input');
    const processingSection = document.getElementById('processing-section');
    const promptTestingSection = document.getElementById('prompt-testing-section');
    const processBtn = document.getElementById('process-ai-btn');
    const testPromptBtn = document.getElementById('test-prompt-btn');
    const clearPromptBtn = document.getElementById('clear-prompt-btn');
    const customPromptTextarea = document.getElementById('custom-prompt');
    const loadingSpinner = document.querySelector('.loading-spinner');
    const loadingText = document.getElementById('loading-text');
    const resultsSection = document.getElementById('results-section');
    const resultsTbody = document.getElementById('results-tbody');
    const promptResults = document.getElementById('prompt-results');
    const contextResultsBody = document.getElementById('context-results-body');
    const statusMessages = document.getElementById('status-messages');

    // Global variables
    let processedData = [];
    let currentStructuredData = null;
    let csvData = '';  // Store CSV data for XLSX export
    let streamedCSVRows = [];
    let tableInitialized = false;

    // Initialize drag and drop events
    uploadArea.addEventListener('click', () => pdfInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    pdfInput.addEventListener('change', handleFileSelect);

    // Process button event
    processBtn.addEventListener('click', processText);

    // Prompt testing events
    testPromptBtn.addEventListener('click', testCustomPrompt);
    clearPromptBtn.addEventListener('click', clearCustomPrompt);

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
                currentStructuredData = data.data;
                showSuccess(data.message);
                processingSection.style.display = 'block';
                promptTestingSection.style.display = 'block';
                
                // Add event listener for AI processing
                document.getElementById('process-ai-btn').addEventListener('click', processText);
            } else {
                showError(data.error || 'Failed to extract data');
            }
        })
        .catch(error => {
            hideLoading();
            showError('Error uploading file: ' + error.message);
        });
    }

    function processText() {
        if (!currentStructuredData) {
            showError('No structured data to process');
            return;
        }

        showLoading('Starting AI processing...');

        // Try streaming first, fallback to regular processing
        processWithStreaming();
    }

    function processWithStreaming() {
        fetch('/process_stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(currentStructuredData)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Streaming not available');
            }
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            
            function readStream() {
                return reader.read().then(({ done, value }) => {
                    if (done) {
                        hideLoading();
                        return;
                    }
                    
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    
                    // Keep the last incomplete line in buffer
                    buffer = lines.pop() || '';
                    
                    lines.forEach(line => {
                        if (line.trim().startsWith('data: ')) {
                            try {
                                const jsonStr = line.slice(6).trim();
                                if (jsonStr && jsonStr !== '{}') {
                                    console.log('Parsing streaming line:', jsonStr.substring(0, 100) + '...');
                                    const data = JSON.parse(jsonStr);
                                    handleStreamingData(data);
                                }
                            } catch (e) {
                                console.error('Error parsing streaming data:', e, 'Line:', line);
                            }
                        }
                    });
                    
                    return readStream();
                });
            }
            
            return readStream();
        })
        .catch(error => {
            console.error('Streaming failed:', error);
            hideLoading();
            showError('Processing failed: ' + error.message);
        });
    }

    function handleStreamingData(data) {
        console.log('Streaming data received:', data.type, data);
        
        if (data.type === 'header') {
            // Handle CSV header
            streamedCSVRows = [data.content];  // Initialize with header
            console.log('Header received:', data.content);
            if (!tableInitialized) {
                initializeStreamingTable();
                tableInitialized = true;
            }
        } else if (data.type === 'row') {
            // Handle CSV row content
            streamedCSVRows.push(data.content);  // Store raw CSV row
            displayStreamingCSVRow(data.content);
        } else if (data.type === 'complete') {
            hideLoading();
            console.log('🎉 Streaming complete. Total rows:', data.total_rows);
            // Reconstruct CSV data from streamed rows
            csvData = streamedCSVRows.join('\n');
            console.log('📊 CSV data reconstructed, length:', csvData.length);
            console.log('📄 CSV data preview:', csvData.substring(0, 200) + '...');
            console.log('🔧 Calling finalizeStreamingDisplay...');
            finalizeStreamingDisplay(data.cost_summary);
            console.log('✅ finalizeStreamingDisplay completed');
        } else if (data.status === 'error') {
            hideLoading();
            showError('Processing failed: ' + data.error);
        } else {
            console.log('Unknown streaming data type:', data);
        }
    }

    function displayStreamingCSVRow(csvRow) {
        if (!tableInitialized) {
            initializeStreamingTable();
            tableInitialized = true;
        }

        // Parse the CSV row (format: "rowN: field,value,context")
        const colonIndex = csvRow.indexOf(': ');
        if (colonIndex === -1) return;

        const actualCsvData = csvRow.substring(colonIndex + 2);
        
        // Simple CSV parsing for display
        const parts = actualCsvData.split('","');
        if (parts.length >= 3) {
            const field = parts[0].replace(/^"/, '');
            const value = parts[1];
            const context = parts[2].replace(/"$/, '');

            const row = document.createElement('tr');
            row.innerHTML = `
                <td><strong>${field}</strong></td>
                <td>${value}</td>
                <td style="max-width: 400px; word-wrap: break-word;">${context || '<em>No context</em>'}</td>
            `;
            resultsTbody.appendChild(row);
        }
    }

    function initializeStreamingTable() {
        resultsSection.style.display = 'block';
        resultsTbody.innerHTML = '';
        tableInitialized = true;
    }

    function finalizeStreamingDisplay(costSummary = null) {
        console.log('🔧 finalizeStreamingDisplay called with costSummary:', !!costSummary);
        
        // Re-attach export event listeners (remove existing first to prevent duplicates)
        const xlsxBtn = document.getElementById('export-xlsx-btn');
        const csvBtn = document.getElementById('export-csv-btn');
        const jsonBtn = document.getElementById('export-json-btn');
        const pdfBtn = document.getElementById('export-pdf-btn');
        
        console.log('🔍 Looking for export buttons...');
        console.log('XLSX button found:', !!xlsxBtn);
        console.log('CSV button found:', !!csvBtn);
        console.log('JSON button found:', !!jsonBtn);
        console.log('PDF button found:', !!pdfBtn);
        
        if (xlsxBtn) {
            xlsxBtn.removeEventListener('click', exportXlsx);
            xlsxBtn.addEventListener('click', exportXlsx);
            console.log('✅ XLSX export button event listener attached');
            console.log('XLSX button element:', xlsxBtn);
            console.log('csvData available for export:', !!csvData, 'Length:', csvData ? csvData.length : 0);
        } else {
            console.error('❌ XLSX export button not found!');
            // Try to find it with a delay
            setTimeout(() => {
                const delayedBtn = document.getElementById('export-xlsx-btn');
                if (delayedBtn) {
                    console.log('🔄 Found XLSX button after delay, attaching listener...');
                    delayedBtn.addEventListener('click', exportXlsx);
                }
            }, 1000);
        }
        
        if (csvBtn) {
            csvBtn.removeEventListener('click', exportCsv);
            csvBtn.addEventListener('click', exportCsv);
        }
        
        if (jsonBtn) {
            jsonBtn.removeEventListener('click', exportJson);
            jsonBtn.addEventListener('click', exportJson);
        }
        
        if (pdfBtn) {
            pdfBtn.removeEventListener('click', exportPdf);
            pdfBtn.addEventListener('click', exportPdf);
        }
        
        // Add summary with cost information
        const summaryDiv = document.createElement('div');
        summaryDiv.className = 'alert alert-success mt-3';
        
        let costInfo = '';
        if (costSummary && costSummary.total_cost_usd) {
            costInfo = ` | LLM Cost: ${costSummary.total_cost_usd.toFixed(6)} (${costSummary.total_tokens.toLocaleString()} tokens, ${costSummary.api_calls} API calls)`;
        }
        
        summaryDiv.innerHTML = `
            <h6>Processing Complete</h6>
            <p>Successfully processed ${streamedCSVRows.length - 1} data entries${costInfo}</p>
        `;
        
        resultsSection.appendChild(summaryDiv);
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    // Custom prompt testing
    function testCustomPrompt() {
        const customPrompt = customPromptTextarea.value.trim();
        
        if (!customPrompt) {
            showError('Please enter a custom prompt');
            return;
        }

        if (!currentStructuredData) {
            showError('Please upload and extract a PDF first');
            return;
        }

        showLoading('Testing custom prompt...');

        fetch('/test_prompt', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ prompt: customPrompt })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Failed to test prompt');
                });
            }
            return response.json();
        })
        .then(result => {
            hideLoading();
            console.log('Prompt test response received:', result);
            
            if (result.success) {
                console.log('Prompt test successful, displaying results...');
                displayPromptResults(result);
                showSuccess(`Custom prompt tested successfully! Found ${result.total_entities} entities.`);
            } else {
                console.error('Prompt test failed:', result.error);
                showError(result.error || 'Prompt test failed');
            }
        })
        .catch(error => {
            hideLoading();
            showError('Error testing prompt: ' + error.message);
        });
    }

    function clearCustomPrompt() {
        customPromptTextarea.value = '';
        promptResults.style.display = 'none';
    }

    // Display prompt test results
    function displayPromptResults(result) {
        console.log('displayPromptResults called with:', result);
        console.log('contextResultsBody element:', contextResultsBody);
        
        if (contextResultsBody) {
            contextResultsBody.innerHTML = '';
            
            if (result.context_data && result.context_data.length > 0) {
                console.log('Processing', result.context_data.length, 'context items');
                
                result.context_data.forEach((item, index) => {
                    console.log(`Processing item ${index}:`, item);
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td><strong>${item.field || 'Unknown Field'}</strong></td>
                        <td>${item.value || 'Unknown Value'}</td>
                        <td style="max-width: 400px; word-wrap: break-word;">${item.context || '<em>No context</em>'}</td>
                    `;
                    contextResultsBody.appendChild(row);
                    console.log('Row added to table');
                });
                
                if (promptResults) {
                    promptResults.style.display = 'block';
                    promptResults.classList.remove('d-none');
                    console.log('Results table shown');
                    
                    // Display performance metrics if available
                    displayPerformanceMetrics(result.performance_stats);
                    
                    // Attach export listeners for prompt results
                    attachPromptExportListeners(result.context_data);
                    
                    // Scroll to results
                    setTimeout(() => {
                        promptResults.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }, 100);
                } else {
                    console.error('promptResults element not found');
                }
            } else {
                console.log('No context data or empty array:', result.context_data);
                showError('No context data returned from prompt test');
            }
        } else {
            console.error('contextResultsBody element not found');
        }
    }

    // Manual test function for debugging
    window.testExportButton = function() {
        console.log('=== EXPORT BUTTON TEST ===');
        const xlsxBtn = document.getElementById('export-xlsx-btn');
        console.log('Button exists:', !!xlsxBtn);
        console.log('Button element:', xlsxBtn);
        console.log('csvData exists:', !!csvData);
        console.log('csvData length:', csvData ? csvData.length : 'N/A');
        console.log('csvData preview:', csvData ? csvData.substring(0, 100) + '...' : 'No data');
        
        if (xlsxBtn) {
            console.log('Button is visible:', xlsxBtn.offsetParent !== null);
            console.log('Button is disabled:', xlsxBtn.disabled);
            console.log('Manually triggering exportXlsx...');
            exportXlsx();
        }
    };

    // Manual function to fix export buttons
    window.fixExportButtons = function() {
        console.log('🔧 Manually fixing export buttons...');
        
        // Try to reconstruct csvData from the table if it's missing
        if (!csvData) {
            console.log('📊 csvData missing, trying to reconstruct from table...');
            const table = document.querySelector('#results-table tbody');
            if (table) {
                const rows = table.querySelectorAll('tr');
                const csvRows = ['Field,Value,Context']; // Header
                
                rows.forEach(row => {
                    const cells = row.querySelectorAll('td');
                    if (cells.length >= 3) {
                        const field = cells[0].textContent.trim();
                        const value = cells[1].textContent.trim();
                        const context = cells[2].textContent.trim();
                        csvRows.push(`"${field}","${value}","${context}"`);
                    }
                });
                
                csvData = csvRows.join('\n');
                console.log('✅ csvData reconstructed from table, length:', csvData.length);
            }
        }
        
        // Attach event listeners
        const xlsxBtn = document.getElementById('export-xlsx-btn');
        if (xlsxBtn) {
            xlsxBtn.removeEventListener('click', exportXlsx);
            xlsxBtn.addEventListener('click', exportXlsx);
            console.log('✅ XLSX button listener attached');
        }
        
        console.log('🎯 Export buttons fixed! Try clicking Export as XLSX now.');
    };

    // Attach export listeners for prompt results
    function attachPromptExportListeners(contextData) {
        const promptXlsxBtn = document.getElementById('export-prompt-xlsx-btn');
        const promptCsvBtn = document.getElementById('export-prompt-csv-btn');
        
        if (promptXlsxBtn) {
            promptXlsxBtn.removeEventListener('click', exportPromptXlsx);
            promptXlsxBtn.addEventListener('click', () => exportPromptXlsx(contextData));
            console.log('Prompt XLSX export button attached');
        }
        
        if (promptCsvBtn) {
            promptCsvBtn.removeEventListener('click', exportPromptCsv);
            promptCsvBtn.addEventListener('click', () => exportPromptCsv(contextData));
            console.log('Prompt CSV export button attached');
        }
    }

    // Export prompt results as XLSX
    function exportPromptXlsx(contextData) {
        if (!contextData || contextData.length === 0) {
            showError('No prompt test data to export');
            return;
        }

        // Create CSV data from context data
        const csvRows = ['Field,Value,Context'];
        contextData.forEach(item => {
            const field = String(item.field || '').replace(/"/g, '""');
            const value = String(item.value || '').replace(/"/g, '""');
            const context = String(item.context || '').replace(/"/g, '""');
            csvRows.push(`"${field}","${value}","${context}"`);
        });
        
        const promptCsvData = csvRows.join('\n');
        
        showLoading('Generating XLSX file from prompt results...');

        fetch('/export_xlsx', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ csv_data: promptCsvData })
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
            link.download = 'prompt_test_results.xlsx';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);
            
            showSuccess('Prompt test results exported as XLSX!');
        })
        .catch(error => {
            hideLoading();
            showError('Export failed: ' + error.message);
        });
    }

    // Export prompt results as CSV
    function exportPromptCsv(contextData) {
        if (!contextData || contextData.length === 0) {
            showError('No prompt test data to export');
            return;
        }

        // Create CSV data from context data
        const csvRows = ['Field,Value,Context'];
        contextData.forEach(item => {
            const field = String(item.field || '').replace(/"/g, '""');
            const value = String(item.value || '').replace(/"/g, '""');
            const context = String(item.context || '').replace(/"/g, '""');
            csvRows.push(`"${field}","${value}","${context}"`);
        });
        
        const promptCsvData = csvRows.join('\n');

        fetch('/export_csv', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ csv_data: promptCsvData })
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
            link.download = 'prompt_test_results.csv';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);
            
            showSuccess('Prompt test results exported as CSV!');
        })
        .catch(error => {
            showError('Export failed: ' + error.message);
        });
    }

    // Display performance metrics
    function displayPerformanceMetrics(stats) {
        if (!stats || Object.keys(stats).length === 0) {
            console.log('No performance stats available');
            return;
        }

        // Find or create performance metrics container
        let metricsContainer = document.getElementById('performance-metrics');
        if (!metricsContainer) {
            metricsContainer = document.createElement('div');
            metricsContainer.id = 'performance-metrics';
            metricsContainer.className = 'alert alert-info mt-3';
            
            // Insert after the prompt results table
            const promptResults = document.getElementById('prompt-results');
            if (promptResults) {
                promptResults.appendChild(metricsContainer);
            }
        }

        // Create performance metrics HTML
        const costSummary = stats.cost_summary || {};
        const processingTime = stats.processing_time_seconds || 0;
        const localPercent = stats.local_processing_percent || 0;
        const successRate = stats.success_rate_percent || 0;
        const apiCalls = stats.gpt_api_calls || 0;
        const totalCost = costSummary.total_cost_usd || 0;

        metricsContainer.innerHTML = `
            <h6><i class="fas fa-chart-line"></i> Performance Metrics</h6>
            <div class="row">
                <div class="col-md-3">
                    <strong>Processing Time:</strong><br>
                    <span class="text-success">${processingTime.toFixed(2)}s</span>
                </div>
                <div class="col-md-3">
                    <strong>Local Processing:</strong><br>
                    <span class="text-primary">${localPercent.toFixed(1)}%</span>
                </div>
                <div class="col-md-3">
                    <strong>Success Rate:</strong><br>
                    <span class="text-info">${successRate.toFixed(1)}%</span>
                </div>
                <div class="col-md-3">
                    <strong>API Cost:</strong><br>
                    <span class="text-warning">$${totalCost.toFixed(4)}</span>
                </div>
            </div>
            <div class="row mt-2">
                <div class="col-md-6">
                    <small><strong>API Calls:</strong> ${apiCalls} (vs 50+ in old method)</small>
                </div>
                <div class="col-md-6">
                    <small><strong>Method:</strong> ${stats.method || 'optimized'}</small>
                </div>
            </div>
        `;

        console.log('Performance metrics displayed:', stats);
    }

    // Export functions
    function exportXlsx() {
        console.log('exportXlsx called');
        console.log('csvData exists:', !!csvData);
        console.log('csvData length:', csvData ? csvData.length : 'N/A');
        
        if (!csvData) {
            console.error('No csvData available for export');
            showError('No data to export. Please process the document first.');
            return;
        }

        console.log('Starting XLSX export...');
        showLoading('Generating XLSX file...');

        fetch('/export_xlsx', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ csv_data: csvData })
        })
        .then(response => {
            console.log('XLSX export response status:', response.status);
            console.log('XLSX export response ok:', response.ok);
            hideLoading();
            
            if (!response.ok) {
                console.error('XLSX export failed with status:', response.status);
                return response.json().then(data => {
                    console.error('XLSX export error data:', data);
                    throw new Error(data.error || 'Failed to generate XLSX');
                });
            }
            
            console.log('XLSX export successful, processing blob...');
            // Handle binary file download
            return response.blob();
        })
        .then(blob => {
            console.log('XLSX blob received, size:', blob.size);
            // Create and click download link
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'extracted_data_comments.xlsx';
            document.body.appendChild(link);
            console.log('Triggering XLSX download...');
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);
            console.log('XLSX download completed');
        })
        .catch(error => {
            hideLoading();
            showError(error.message);
        });
    }

    function exportCsv() {
        if (!csvData) {
            showError('No data to export');
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
            link.download = 'extracted_data_comments.csv';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);
        })
        .catch(error => {
            showError(error.message);
        });
    }

    function exportJson() {
        if (!processedData.length) {
            showError('No data to export');
            return;
        }

        const jsonData = JSON.stringify(processedData, null, 2);
        const blob = new Blob([jsonData], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'extracted_data.json';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
    }

    function exportPdf() {
        if (!processedData.length) {
            showError('No data to export');
            return;
        }

        showError('PDF export not implemented yet');
    }

    // Utility functions
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

    // Show raw JSON data modal
    window.showJsonModal = function() {
        if (currentStructuredData) {
            document.getElementById('json-content').textContent = JSON.stringify(currentStructuredData, null, 2);
            new bootstrap.Modal(document.getElementById('jsonModal')).show();
        } else {
            showError('No data to display');
        }
    };
});