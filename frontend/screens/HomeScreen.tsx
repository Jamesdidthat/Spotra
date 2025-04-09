import { CameraView } from "expo-camera";
import { useEffect, useRef, useState } from "react";
import { View, TouchableOpacity, Text, StyleSheet, Image, Alert } from "react-native";

const SERVER_URL = "http://192.168.81.19:5000"; // Replace with your local server IP

const HomeScreen = () => {
  const cameraRef = useRef<CameraView>(null);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [imageUri, setImageUri] = useState<string | null>(null);

  const captureFrame = async () => {
    if (!cameraRef.current) return;

    try {
      const photo = await cameraRef.current.takePictureAsync({ base64: true });
      if (photo?.uri) {
        setImageUri(photo.uri); // Display captured image
      }

      const formData = new FormData();
      if (!photo) return;

      formData.append("file", {
        uri: photo.uri,
        name: "frame.jpg",
        type: "image/jpeg",
      } as any);

      const response = await fetch(`http://192.168.81.19:5000/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      if (data.error) {
        Alert.alert("Prediction Error", data.error);
      } else {
        setPrediction(`${data.movie} (${data.confidence.toFixed(2)}%)`);
      }
    } catch (error) {
      console.error("Prediction Error:", error);
      Alert.alert("Error", "Failed to send image. Check your network & server.");
    }
  };

  return (
    <View style={styles.container}>
      <CameraView style={styles.camera} ref={cameraRef}>
        {imageUri && <Image source={{ uri: imageUri }} style={styles.preview} />}
        <TouchableOpacity style={styles.button} onPress={captureFrame}>
          <Text style={styles.text}>Capture & Predict</Text>
        </TouchableOpacity>
        {prediction && <Text style={styles.prediction}>{prediction}</Text>}
      </CameraView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: "center" },
  camera: { flex: 1 },
  button: {
    alignSelf: "center",
    margin: 20,
    backgroundColor: "black",
    padding: 10,
    borderRadius: 10,
  },
  text: { color: "white", fontSize: 18 },
  preview: {
    width: 100,
    height: 100,
    position: "absolute",
    bottom: 20,
    right: 20,
    borderRadius: 10,
  },
  prediction: { color: "yellow", fontSize: 18, textAlign: "center", marginTop: 10 },
});

export default HomeScreen;
