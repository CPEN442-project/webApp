
const mongoose = require('mongoose');
const Joi = require('joi');
const express = require('express');
const app = express();
// Example backend code for JWT authentication
const jwt = require('jsonwebtoken');
// Example backend code for WebSocket
const WebSocket = require('ws');


var isTest = true;
exports.isTest = isTest;

var mongoURI = null
if (isTest) {
  mongoURI = 'mongodb://localhost:27017/test_webapp';
} else {
  // For actual project deployment
  mongoURI = 'mongodb://localhost:27017/webapp';
}

// Create connection for calendoDB
const db = mongoose.createConnection(mongoURI, { useNewUrlParser: true, useUnifiedTopology: true });


// Store data in app.locals
app.locals.mongoDB = db;



// Create an HTTPS server with the SSL certificate and private key
const httpsOptions = {
  key: fs.readFileSync('path/to/private-key.pem'), // Replace with the path to your private key file
  cert: fs.readFileSync('path/to/certificate.pem'), // Replace with the path to your certificate file
};

const httpsServer = https.createServer(httpsOptions, app);



// Secret key for signing JWTs
const secretKey = 'yourSecretKey';

// Function to generate a JWT token
function generateToken(userId) {
  return jwt.sign({ userId }, secretKey, { expiresIn: '1h' });
}

// Middleware to verify JWT in incoming requests
function verifyToken(req, res, next) {
  const token = req.headers.authorization;
  if (!token) {
    return res.status(401).json({ message: 'Unauthorized' });
  }

  jwt.verify(token, secretKey, (err, decoded) => {
    if (err) {
      return res.status(401).json({ message: 'Unauthorized' });
    }
    req.userId = decoded.userId;
    next();
  });
}

wss.on('connection', (ws) => {
  console.log('WebSocket connection established');

  ws.on('message', (message) => {
    const data = JSON.parse(message);
    handleWebSocketMessage(ws, data);
  });

  ws.on('close', () => {
    console.log('WebSocket connection closed');
    // Perform cleanup or additional actions on session end
  });
});

function handleWebSocketMessage(ws, data) {
  // Handle different types of WebSocket messages
  switch (data.type) {
    case 'keystroke':
      // Process keystroke data
      console.log(`Received keystroke: ${data.keystroke} from user ${data.userId}`);
      break;
    // Add more cases for different message types
  }
}


class AnomalyDetector {
  constructor() {
    this.userProfiles = new Map();
  }

  processEvent(event) {
    if (event.event === 'key') {
      this.updateUserProfile(event);
    } else if (event.event === 'click') {
      // Process mouse click event
      console.log(`Mouse click at position: ${event.position}`);
    }
  }

  updateUserProfile(event) {
    const { timestamp, input } = event;
    const userId = 'exampleUserId'; // Replace with actual user identification logic

    if (!this.userProfiles.has(userId)) {
      this.userProfiles.set(userId, {
        keystrokes: [],
        lastTimestamp: timestamp,
      });
      return;
    }

    const userProfile = this.userProfiles.get(userId);
    const timeDifference = timestamp - userProfile.lastTimestamp;

    // Calculate Characters Per Second (CPS) based on time difference and number of characters
    const charactersTyped = input.length;
    const charactersPerSecond = charactersTyped / (timeDifference / 1000);

    // Detect anomaly based on CPS threshold (adjust threshold as needed)
    const anomalyThreshold = 0.1; // Example threshold, adjust as needed
    if (charactersPerSecond < anomalyThreshold) {
      console.log(`Potential anomaly detected for user ${userId}. Low CPS: ${charactersPerSecond}`);
      // Trigger further actions or alerts as needed
    }

    // Update user profile for continuous learning
    userProfile.keystrokes.push({ timestamp, input });
    userProfile.lastTimestamp = timestamp;

    // Log the calculated CPS
    console.log(`CPS for user ${userId}: ${charactersPerSecond}`);
  }
}

// Example usage
const anomalyDetector = new AnomalyDetector();

// Example input array
const inputEvents = [
  { event: 'key', timestamp: 1636600000000, input: 'a' },
  { event: 'key', timestamp: 1636600010000, input: 'b' },
  { event: 'click', timestamp: 1636600020000, position: { x: 10, y: 20 } },
  // Add more events as needed
];

// Process each event in the input array
inputEvents.forEach(event => anomalyDetector.processEvent(event));


// Example secure WebSocket connection
const wss = new WebSocket.Server({
  server: httpsServer, // Your HTTPS server
  /* other options */
});


// Start the HTTPS server
const port = process.env.PORT || 3000;
const host = 'localhost';

httpsServer.listen(port, () => {
  console.log(`Server is running on https://${host}:${port}`);
});