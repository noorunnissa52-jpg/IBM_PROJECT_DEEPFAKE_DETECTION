const imageInput = document.getElementById('imageInput');
const preview = document.getElementById('preview');
const predictBtn = document.getElementById('predictBtn');
const uploadText = document.getElementById('uploadText');
const resultBox = document.getElementById('resultBox');
const label = document.getElementById('label');
const confidence = document.getElementById('confidence');
const probabilityList = document.getElementById('probabilityList');

let selectedFile = null;

imageInput.addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (!file) return;

  selectedFile = file;
  uploadText.textContent = file.name;

  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.hidden = false;
  };
  reader.readAsDataURL(file);
});

predictBtn.addEventListener('click', async () => {
  if (!selectedFile) {
    alert('Please choose an image first.');
    return;
  }

  const formData = new FormData();
  formData.append('file', selectedFile);

  predictBtn.disabled = true;
  predictBtn.textContent = 'Predicting...';

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const err = await response.text();
      throw new Error(err || 'Prediction failed');
    }

    const payload = await response.json();
    const { labelName, confidenceValue, probabilities } = payload;

    label.textContent = labelName;
    label.className = `label ${labelName.toLowerCase()}`;
    confidence.textContent = `Confidence: ${(confidenceValue * 100).toFixed(1)}%`;

    probabilityList.innerHTML = '';
    Object.entries(probabilities).forEach(([key, value]) => {
      const item = document.createElement('div');
      item.className = 'probability-item';
      item.innerHTML = `<span>${key}</span><span>${(value * 100).toFixed(1)}%</span>`;
      probabilityList.appendChild(item);
    });

    resultBox.classList.remove('hidden');
  } catch (error) {
    alert(error.message || 'Could not make a prediction.');
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = 'Predict';
  }
});
