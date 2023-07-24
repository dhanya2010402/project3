

const express = require('express');
const multer = require('multer');
const fs = require('fs');
const { faceRecognition } = require('./faceRecognition');
const cors = require('cors');
const path = require('path');
const faceapi = require('face-api.js');
const canvas = require('canvas');
const { Canvas, Image, ImageData } = canvas;

faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const app = express();
const upload = multer({
  dest: 'uploads/',
  filename: (req, file, cb) => {
    const fileName = file.originalname;
    cb(null, fileName);
  },
});

app.use(cors());
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Load models before handling requests
Promise.all([
  faceapi.nets.ssdMobilenetv1.loadFromDisk('models'),
  faceapi.nets.faceLandmark68Net.loadFromDisk('models'),
  faceapi.nets.faceRecognitionNet.loadFromDisk('models'),
  faceapi.nets.faceExpressionNet.loadFromDisk('models/faceExpressionNet'),
 
])
  .then(() => {
    console.log('Models loaded');
    loadKnownFaces(); // Load known faces after models are loaded
  })
  .catch((error) => {
    console.error('Error loading models:', error);
  });

// Array to store known faces with their descriptors
const knownFaces = [];

// Function to load known faces with their descriptors
const loadKnownFaces = async () => {
  const files = fs.readdirSync('known_faces');

  for (const file of files) {
    const imgPath = path.join('known_faces', file);
    const img = await canvas.loadImage(imgPath);
    
    // Load the models if they haven't been loaded before
    if (!faceapi.nets.ssdMobilenetv1.params) {
      await faceapi.nets.ssdMobilenetv1.loadFromDisk('models');
    }

    if (!faceapi.nets.faceLandmark68Net.params) {
      await faceapi.nets.faceLandmark68Net.loadFromDisk('models');
    }

    if (!faceapi.nets.faceRecognitionNet.params) {
      await faceapi.nets.faceRecognitionNet.loadFromDisk('models');
    }

    if (!faceapi.nets.faceExpressionNet.params) {
      await faceapi.nets.faceExpressionNet.loadFromDisk('models/faceExpressionNet');
    }

    const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();

    if (detections) {
      const descriptor = detections.descriptor;
      knownFaces.push({ name: file.split('.')[0], descriptor });
    } else {
      console.error('Failed to extract descriptor for:', file);
    }
  }

  console.log('Known faces loaded');
};


// Face Detection route
app.post('/face-detection', upload.single('image'), async (req, res) => {
  console.log(req.file);

  const imageUrl = `${req.protocol}://${req.get('host')}/uploads/${req.file.filename}`;

  try {
    // Load the input image
    const img = await canvas.loadImage(imageUrl);

    // Detect faces in the image
    const detections = await faceapi
      .detectAllFaces(img)
      .withFaceLandmarks()
      .withFaceDescriptors();

    // Draw bounding boxes around the detected faces
    const outImg = faceapi.createCanvasFromMedia(img);
    faceapi.draw.drawDetections(outImg, detections);

    // Convert the modified image to base64 string
    const modifiedImgDataUrl = outImg.toDataURL();

    res.status(200).json({ imageUrl: modifiedImgDataUrl });
    console.log('Modified image sent to client');
  } catch (error) {
    console.error('Error processing image:', error);
    res.status(500).json({ message: 'Error processing image' });
  }
});

// Facial Expression route
app.post('/facial-expression', upload.single('image'), async (req, res) => {
  console.log(req.file);

  const imageUrl = `${req.protocol}://${req.get('host')}/uploads/${req.file.filename}`;

  try {
    // Load the input image
    const img = await canvas.loadImage(imageUrl);

    // Detect faces and facial expressions in the image
    const detections = await faceapi
      .detectAllFaces(img)
      .withFaceLandmarks()
      .withFaceExpressions();

    // Draw facial expressions on the detected faces
    const outImg = faceapi.createCanvasFromMedia(img);
    faceapi.draw.drawFaceExpressions(outImg, detections);

    // Convert the modified image to base64 string
    const modifiedImgDataUrl = outImg.toDataURL();

    // Get the detected facial expressions
    const expressions = detections.map((detection) => detection.expressions.asSortedArray()[0].expression);

    res.status(200).json({ imageUrl: modifiedImgDataUrl, expressions });
    console.log('Modified image and expressions sent to client');
  } catch (error) {
    console.error('Error processing image:', error);
    res.status(500).json({ message: 'Error processing image' });
  }
});
app.post('/face-recognition', upload.single('image'), async (req, res) => {
  console.log(req.file);

  const imageUrl = `${req.protocol}://${req.get('host')}/uploads/${req.file.filename}`;

  try {
    // Load the input image
    const img = await canvas.loadImage(imageUrl);

    // Detect faces in the image
    const detections = await faceapi
      .detectAllFaces(img)
      .withFaceLandmarks()
      .withFaceDescriptors();

    if (detections.length === 0) {
      // No faces detected
      res.status(200).json({ name: 'unknown', imageUrl });
      return;
    }

    // Recognize the detected faces
    const recognizedFaces = detections.map((detection) => {
      const faceDescriptor = detection.descriptor;
      const bestMatch = faceRecognition(faceDescriptor, knownFaces);
      return { name: bestMatch.name, distance: bestMatch.distance };
    });

    // Sort recognized faces by distance (lower distance means better match)
    recognizedFaces.sort((a, b) => a.distance - b.distance);

    const bestMatch = recognizedFaces[0];
    if (bestMatch.distance <= 0.5) {
      // Recognized as known person
      res.status(200).json({ name: bestMatch.name, imageUrl });
    } else {
      // Not recognized as known person
      res.status(200).json({ name: 'unknown', imageUrl });
    }

    console.log('Recognition result sent to client');
  } catch (error) {
    console.error('Error processing image:', error);
    res.status(500).json({ message: 'Error processing image' });
  }
});
app.post('/face-similarity', upload.array('image', 2), async (req, res) => {
  const imageUrls = req.files.map((file) => `${req.protocol}://${req.get('host')}/uploads/${file.filename}`);

  try {
    console.log("images received:", imageUrls);

    // Load the input images
    const img1 = await canvas.loadImage(imageUrls[0]);
    const img2 = await canvas.loadImage(imageUrls[1]);

    // Detect faces and get their descriptors for both images
    const detections1 = await faceapi.detectSingleFace(img1).withFaceLandmarks().withFaceDescriptor();
    const detections2 = await faceapi.detectSingleFace(img2).withFaceLandmarks().withFaceDescriptor();
    
    if (!detections1 || !detections2) {
      // No faces detected in one or both images
      res.status(400).json({ message: 'No faces detected in one or both images' });
      return;
    }

    // Calculate the Euclidean distance between the descriptors
    const distance = faceapi.euclideanDistance(detections1.descriptor, detections2.descriptor);
    console.log("similarity distance:", distance);

    res.status(200).json({ imageUrl1: imageUrls[0], imageUrl2: imageUrls[1], distance });
    console.log('Face similarity result sent to client');
  } catch (error) {
    console.error('Error processing images:', error);
    res.status(500).json({ message: 'Error processing images' });
  }
});


app.listen(8000, () => {
  console.log('Server is running on port 8000');
});