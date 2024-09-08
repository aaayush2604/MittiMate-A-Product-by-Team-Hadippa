import React, { useState } from "react";

const AppointmentForm = () => {
  const [message, setMessage] = useState("");

  const handleSubmit = () => {
    setMessage("Your request has been received! We'll connect with you shortly.");
  };

  return (
    <div className="background">
      <div className="shape" id="fcircle"></div>
      <div className="shape" id="scircle"></div>
      <section className="container">
        <header>Appointment Form</header>
        <form className="form" id="appointmentForm">
          <div className="input-box">
            <label>Full Name</label>
            <input type="text" placeholder="Enter full name" required />
          </div>

          <div className="input-box">
            <label>Email Address</label>
            <input type="email" placeholder="Enter email address" required />
          </div>

          <div className="column">
            <div className="input-box">
              <label>Phone Number</label>
              <input type="number" placeholder="Enter phone number" required />
            </div>
            <div className="input-box">
              <label>Preferred Appointment Date</label>
              <input type="date" required />
            </div>
          </div>

          <div className="time-box">
            <h3>Preferred Time Slot</h3>
            <div className="time-option">
              <div className="time">
                <input type="radio" id="time1" name="time" defaultChecked />
                <label htmlFor="time1">Morning: 9.00 a.m. to 11.00 a.m.</label>
              </div>
              <div className="time">
                <input type="radio" id="time2" name="time" />
                <label htmlFor="time2">Afternoon: 12.00 p.m. to 2.00 p.m.</label>
              </div>
              <div className="time">
                <input type="radio" id="time3" name="time" />
                <label htmlFor="time3">Evening: 3.00 p.m. to 5.00 p.m.</label>
              </div>
            </div>
          </div>

          <div className="input-box address">
            <label>Address</label>
            <input type="text" placeholder="Enter street address" required />
            <input type="text" placeholder="Enter street address line 2" />
            <div className="column">
              <div className="select-box">
                <select>
                  <option hidden>Country</option>
                  <option>India</option>
                </select>
              </div>
              <input type="text" placeholder="Enter your State" required />
            </div>
            <div className="column">
              <input type="text" placeholder="Enter your City" required />
              <input type="number" placeholder="Enter postal code" required />
            </div>
          </div>

          <button type="button" onClick={handleSubmit}>Submit</button>
          {message && <span id="message">{message}</span>}
        </form>
      </section>
    </div>
  );
};

export default AppointmentForm;