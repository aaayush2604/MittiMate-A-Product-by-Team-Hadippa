import React from "react";
import ProfilePic from "../Assets/farmer-image-1.jpg";
import { AiFillStar } from "react-icons/ai";

const Testimonial = () => {
  return (
    <div className="work-section-wrapper">
      <div className="work-section-top">
        <p className="primary-subheading">Testimonial</p>
        <h1 className="primary-heading">What They Are Saying</h1>
        <p className="primary-text">
        Hear from Farmers Who've Transformed Their Harvests
        </p>
      </div>
      <div className="testimonial-section-bottom">
        <img src={ProfilePic} alt="" width="100vu" />
        <p>
        यह वेब एप्लिकेशन मेरे उर्वरक प्रबंधन को पूरी तरह से बदल दिया है। मेरे मिट्टी और फसल की जरूरतों के अनुसार की गई सिफारिशों के साथ, मैंने बेहतर उपज और लागत में बचत देखी है। यह एक बेहतरीन उपकरण है सतत खेती के लिए!
        </p>
        <div className="testimonials-stars-container">
          <AiFillStar />
          <AiFillStar />
          <AiFillStar />
          <AiFillStar />
          <AiFillStar />
        </div>
        <h2>— रवि पटेल, किसान</h2>
      </div>
    </div>
  );
};

export default Testimonial;