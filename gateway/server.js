const express = require('express');
const cors = require('cors');
const axios = require('axios');
const multer = require('multer');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5000;
const FASTAPI_URL = process.env.FASTAPI_URL || 'http://localhost:5001';

// Multer handles multipart/form-data in memory
const upload = multer({ storage: multer.memoryStorage() });

app.use(cors());
app.use(express.json());

// Health check for the gateway itself
app.get('/health', (req, res) => {
    res.json({ success: true, service: 'vision-gateway', status: 'up' });
});

// Proxy endpoint for image classification
app.post('/api/classify', upload.single('file'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ success: false, error: 'No file uploaded' });
    }

    try {
        // Create a new FormData to forward to FastAPI
        const formData = new FormData();
        const blob = new Blob([req.file.buffer], { type: req.file.mimetype });
        formData.append('file', blob, req.file.originalname);

        const response = await axios.post(`${FASTAPI_URL}/classify`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });
        res.json(response.data);
    } catch (error) {
        console.error('Proxy error:', error.message);
        res.status(500).json({ 
            success: false, 
            error: 'Failed to reach vision API' 
        });
    }
});

app.listen(PORT, () => {
    console.log(`====================================================`);
    console.log(` EXPRESS VISION GATEWAY  v1.0`);
    console.log(`====================================================`);
    console.log(` Port: ${PORT}     URL: http://localhost:${PORT}`);
    console.log(` Proxying /api/classify -> ${FASTAPI_URL}/classify`);
    console.log(`====================================================`);
});