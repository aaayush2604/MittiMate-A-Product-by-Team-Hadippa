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
        <header>Create your account...</header>
        <form className="form" id="recommendationForm">
          <div className="input-box">
            <label>Full Name</label>
            <input type="text" placeholder="Enter full name" required />
          </div>

          <div className="input-box">
            <label>Email Address(Optional)</label>
            <input type="email" placeholder="Enter email address" required />
          </div>

          <div className="input-box">
            <label>Phone Number</label>
            <input type="number" placeholder="Enter phone number" required />
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

          <button type="button" onClick={handleSubmit}>Create Account</button>
          <Link to="/signin"><button className="secondary-button,button123">Already have an account?</button></Link>
          {message && <span id="message">{message}</span>}
        </form>
      </section>
    </div>
  );
};

export default RecommendationForm;