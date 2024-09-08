import React, { useState } from "react";
import "./form.css";

const RecommendationForm = () => {
  const [message, setMessage] = useState("");

  const handleSubmit = () => {
    setMessage("Your request has been received! We'll connect with you shortly.");
  };

  return (
    <div className="background">
      <div className="shape" id="fcircle"></div>
      <div className="shape" id="scircle"></div>
      <section className="container">
        <header>Recommendation Form</header>
        <form className="form" id="recommendationForm">
          <div className="input-box">
            <label>Full Name</label>
            <input type="text" placeholder="Enter full name" required />
          </div>

          <div className="input-box">
            <label>Email Address</label>
            <input type="email" placeholder="Enter email address" required />
          </div>

          <div className="input-box">
            <label>Phone Number</label>
            <input type="number" placeholder="Enter phone number" required />
          </div>

          <div className="column">
            <div className="input-box">
              <label>Nitrogen Level (mg/kg)</label>
              <input type="number" placeholder="Enter Crop Nitrogen level" required />
            </div>
            <div className="input-box">
              <label>Phosphorus Level (mg/kg)</label>
              <input type="number" placeholder="Enter Crop Phosphorus level" required />
            </div>
          </div>

          <div className="column">
            <div className="input-box">
              <label>Potassium Level (mg/kg)</label>
              <input type="number" placeholder="Enter Crop Potassium level" required />
            </div>
            <div className="input-box">
              <label>Soil pH Level</label>
              <input type="number" placeholder="Enter Soil pH level" required />
            </div>
          </div>

          <div className="column">
            <div className="input-box">
              <label>Rainfall (mm)</label>
              <input type="number" placeholder="Enter length of rainfall received" required />
            </div>
            <div className="input-box">
              <label>Average Temperature (°C)</label>
              <input type="number" placeholder="Enter Temperature" required />
            </div>
          </div>

          <div className="input-box">
            <label>Crop Type</label>
            <input type="text" placeholder="Enter type of crop grown" required />
          </div>

          <div className="column">
            <div className="input-box">
              <label>District</label>
              <input type="text" placeholder="Enter your District" required />
            </div>
            <div className="input-box">
              <label>State</label>
              <input type="text" placeholder="Enter your State" required />
            </div>
          </div>

          <button type="button" onClick={handleSubmit}>Submit</button>
          {message && <span id="message">{message}</span>}
        </form>
      </section>
    </div>
  );
};

export default RecommendationForm;
