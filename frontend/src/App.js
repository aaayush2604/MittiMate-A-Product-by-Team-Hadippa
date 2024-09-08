import React from "react";
import { Routes, Route } from "react-router-dom";
import Home from './Components/Home';
import About from './Components/About';
import Work from './Components/Work';
import Testimonial from './Components/Testimonial';
import Contact from './Components/Contact';
import Footer from './Components/Footer';
import Navbar from './Components/Navbar';
import Form from './Components/AppointmentForm'; // Assuming you will create a Form page
import PlusTest from './Components/RecommendationForm'; // Assuming you will create a PlusTest page
import SignUp from './Components/SignUp';
import SignIn from './Components/SignIn';
import './App.css';

function App() {
  return (
      <div className="App">
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
          <Route path="/work" element={<Work />} />
          <Route path="/testimonial" element={<Testimonial />} />
          <Route path="/contact" element={<Contact />} />
          <Route path="/form" element={<Form />} />
          <Route path="/plustest" element={<PlusTest />} />
          <Route path="/signup" element={<SignUp />} />
          <Route path="/signin" element={<SignIn />} />

        </Routes>
        <Footer />
      </div>
  );
}

export default App;
