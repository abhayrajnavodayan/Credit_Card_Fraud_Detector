document.addEventListener('DOMContentLoaded', function () {
    const predictButton = document.getElementById('predict-button');
    const resultContainer = document.getElementById('result');
    const predictionForm = document.getElementById('prediction-form');

    predictButton.addEventListener('click', function () {
        const formData = new FormData(predictionForm);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            const predictionResult = data.result;

            // Display the result
            resultContainer.innerHTML = `<h2>Result:</h2><p>${predictionResult}</p>`;
            resultContainer.style.display = 'block';

            // Hide the result after 3 seconds
            setTimeout(function () {
                resultContainer.style.display = 'none';
            }, 7000);
        })
        .catch(error => console.error('Error:', error));
    });
});
