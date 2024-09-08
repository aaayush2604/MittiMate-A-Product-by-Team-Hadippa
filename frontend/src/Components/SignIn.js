import React, { useState } from "react";
import { Link } from "react-router-dom"; 

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
        <header>Log in to your account...</header>
        <form className="form" id="recommendationForm">

          <div className="input-box">
            <label>Phone Number</label>
            <input type="number" placeholder="Enter phone number" required />
          </div>

          <div className="column">
            <div className="input-box">
              <label>OTP</label>
              <input type="text" placeholder="Enter One Time Password" required />
            </div>
            
            
          </div>

          <button type="button" onClick={handleSubmit}>Send OTP</button>
          <Link to="/signin"><button className="secondary-button,button123">Verify OTP and Sign In</button></Link>
          {message && <span id="message">{message}</span>}
        </form>
      </section>
    </div>
  );
};

export default RecommendationForm;