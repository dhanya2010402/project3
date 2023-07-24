const faceapi = require('face-api.js');

const faceRecognition = (targetDescriptor, knownFaces) => {
  let bestMatch = { name: 'unknown', distance: 1 };

  knownFaces.forEach((face) => {
    const distance = faceapi.euclideanDistance(targetDescriptor, face.descriptor);

    if (distance < bestMatch.distance) {
      bestMatch = { name: face.name, distance };
    }
  });

  return bestMatch;
};

module.exports = { faceRecognition };
