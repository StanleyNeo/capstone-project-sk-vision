import { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResult(null);
      setError('');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setError('');
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Connect to the Express Gateway on port 5000
      const response = await axios.post('http://localhost:5000/api/classify', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(response.data.data);
    } catch (err) {
      console.error(err);
      setError('Failed to classify image. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>CIFAR-10 Image Classifier</h1>
      <p>Upload an image of an object (plane, car, bird, cat, deer, dog, frog, horse, ship, truck).</p>

      <form onSubmit={handleSubmit}>
        <input 
          type="file" 
          accept="image/*" 
          onChange={handleFileChange} 
          className="file-input"
        />
        <br />
        {preview && <img src={preview} alt="Preview" className="preview-img" />}
        <br />
        <button type="submit" disabled={loading || !file}>
          {loading ? 'Classifying...' : 'Classify Image'}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {result && (
        <div className="result-card">
          <h2>Prediction: <span className="prediction">{result.label.toUpperCase()}</span></h2>
          <p>Confidence: {(result.confidence * 100).toFixed(2)}%</p>
          
          <div className="confidence-bar-container">
            <div
              className="confidence-bar"
              style={{ width: `${result.confidence * 100}%` }}
            ></div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;