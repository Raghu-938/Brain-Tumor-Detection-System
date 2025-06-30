$(document).ready(function () {
    $('#loading, #result, #accuracyLoading, #metrics, #datasetUploadStatus').hide();

    $('#uploadForm').submit(function (e) {
        e.preventDefault();
        $('#loading').show();
        $('#result').hide();

        var formData = new FormData(this);

        $.ajax({
            url: '/predict',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function (data) {
                $('#loading').hide();
                $('#result').show();
                if (data.result) {
                    $('#prediction').text(data.result);
                    $('#originalImage').attr('src', data.original_image);
                    $('#processedImage').attr('src', data.processed_image);
                    $('#downloadButton').attr('href', data.processed_image);
                } else {
                    alert('Error: ' + data.error);
                }
            },
            error: function (error) {
                $('#loading').hide();
                alert('Error: ' + (error.responseJSON?.error || error.responseText));
            }
        });
    });

    $('#datasetUploadForm').submit(function(e) {
        e.preventDefault();
        
        var files = $('#datasetFiles')[0].files;
        if (files.length === 0) {
            alert('Please select at least one file');
            return;
        }

        var validExtensions = ['.zip', '.jpg', '.jpeg', '.png'];
        var maxSize = 500 * 1024 * 1024; // 500MB
        var totalSize = 0;

        for (var i = 0; i < files.length; i++) {
            var file = files[i];
            var fileName = file.name.toLowerCase();
            var isValid = validExtensions.some(ext => fileName.endsWith(ext));
            
            if (!isValid) {
                alert('Only ZIP, JPG/JPEG, and PNG files are allowed\nInvalid file: ' + file.name);
                return false;
            }
            
            totalSize += file.size;
            if (totalSize > maxSize) {
                alert('Total upload size exceeds 500MB limit');
                return false;
            }
        }
        $('#datasetLoading').show();
        $('#datasetResult').hide();
        $('#datasetProgress').css('width', '0%');
        
        var formData = new FormData(this);
        
        $.ajax({
            url: '/load_dataset',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            xhr: function() {
                var xhr = new window.XMLHttpRequest();
                xhr.upload.addEventListener('progress', function(e) {
                    if (e.lengthComputable) {
                        var percent = Math.round((e.loaded / e.total) * 100);
                        $('#datasetProgress').css('width', percent + '%');
                    }
                }, false);
                return xhr;
            },
            success: function(response) {
                $('#datasetLoading').hide();
                
                var resultHtml = `<p>${response.message}</p>
                                <div class="alert alert-success">
                                    <strong>Dataset Statistics:</strong>
                                    <ul>
                                        <li>Total Images: ${response.stats.total_images}</li>
                                        <li>With Tumor: ${response.stats.with_tumor}</li>
                                        <li>Without Tumor: ${response.stats.without_tumor}</li>
                                    </ul>
                                </div>`;
                
                if (response.warnings) {
                    resultHtml += `<div class="alert alert-warning">${response.warnings}</div>`;
                }
                
                $('#datasetMessage').html(resultHtml);
                $('#datasetResult').show();
            },
            error: function(xhr) {
                $('#datasetLoading').hide();
                
                var errorMsg = 'Error uploading dataset';
                try {
                    if (xhr.responseJSON && xhr.responseJSON.error) {
                        errorMsg = xhr.responseJSON.error;
                    } else if (xhr.statusText) {
                        errorMsg = xhr.statusText;
                    }
                } catch (e) {
                    console.error('Error parsing error response:', e);
                }
                
                $('#datasetMessage').html(`<div class="alert alert-danger">${errorMsg}</div>`);
                $('#datasetResult').show();
            }
        });
    });

    $('#preprocessBtn').click(function () {
        $('#preprocessStatus').text('Preprocessing...');
        $.ajax({
            url: '/preprocess_dataset',
            type: 'GET',
            success: function (data) {
                $('#preprocessStatus').text(data.message);
            },
            error: function (error) {
                $('#preprocessStatus').text('Error: ' + (error.responseJSON?.error || 'Failed to preprocess'));
            }
        });
    });

    $('#fetchAccuracyBtn').click(function () {
        $('#accuracyLoading').show();
        $('#metrics').hide();

        $.ajax({
            url: '/accuracy',
            type: 'GET',
            success: function (data) {
                $('#accuracyLoading').hide();
                $('#metrics').show();
                $('#acc').text(data.accuracy);
                $('#prec').text(data.precision);
                $('#rec').text(data.recall);
                $('#f1').text(data.f1_score);
            },
            error: function (error) {
                $('#accuracyLoading').hide();
                alert('Error: ' + (error.responseJSON?.error || error.responseText));
            }
        });
    });
});
