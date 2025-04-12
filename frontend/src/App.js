import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResult(response.data);
    } catch (error) {
      setResult({ error: error.response?.data?.error || 'Failed to analyze' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <h1>Oral Cancer Detection</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => setFile(e.target.files[0])}
        />
        <button type="submit" disabled={!file || loading}>
          {loading ? 'Analyzing...' : 'Analyze Image'}
        </button>
      </form>

      {result && (
        <div className="result">
          {result.error ? (
            <p className="error">{result.error}</p>
          ) : (
            <>
              <h2>Results:</h2>
              <p>Diagnosis: <strong>{result.diagnosis}</strong></p>
              <p>Confidence: <strong>{result.confidence}%</strong></p>
              <p>Raw Prediction: {result.prediction}</p>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default App;