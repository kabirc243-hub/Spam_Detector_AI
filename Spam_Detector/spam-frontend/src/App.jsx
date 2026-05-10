import { useState } from "react";

function App() {

  const [message, setMessage] = useState("");
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState("");
  const [loading, setLoading] = useState(false);

  const checkSpam = async () => {

    if (message.trim() === "") {
      alert("Please enter a message");
      return;
    }

    setLoading(true);

    try {

      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          message: message
        })
      });

      const data = await response.json();

      setPrediction(data.prediction);
      setConfidence(data.confidence);

    } catch (error) {

      console.error(error);
      alert("Backend connection failed");

    }

    setLoading(false);
  };

  return (

    <div style={{
      display: "flex",
      flexDirection: "column",
      justifyContent: "center",
      alignItems: "center",
      height: "100vh",
      fontFamily: "Arial"
    }}>

      <h1>Spam Detection App</h1>

      <textarea
        rows="6"
        cols="50"
        placeholder="Enter message..."
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        style={{
          padding: "10px",
          fontSize: "16px"
        }}
      />

      <br />

      <button
        onClick={checkSpam}
        style={{
          padding: "10px 20px",
          fontSize: "16px",
          cursor: "pointer"
        }}
      >
        {loading ? "Checking..." : "Detect Spam"}
      </button>

      <br />

      {prediction && (
        <h2>
          Result:
          <span
            style={{
              color: prediction === "Spam" ? "red" : "green",
              marginLeft: "10px"
            }}
          >
            {prediction}
          </span>

          <div>
            Confidence: {confidence}%
          </div>
        </h2>
      )}

    </div>
  );
}

export default App;