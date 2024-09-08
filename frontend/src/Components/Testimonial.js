import React from "react";
import ProfilePic1 from "../Assets/farmer-image-1.jpg";
import ProfilePic2 from "../Assets/farmer-image-2.jpg";
import ProfilePic3 from "../Assets/farmer-image-3.jpg";
import ProfilePic4 from "../Assets/farmer-image-4.jpg";
import ProfilePic5 from "../Assets/farmer-image-5.jpg";
import ProfilePic6 from "../Assets/farmer-image-6.jpg";
import ProfilePic7 from "../Assets/farmer-image-7.jpg";
import { AiFillStar } from "react-icons/ai";

const Testimonial = () => {
  const testimonials = [
    {
      img: ProfilePic1,
      text: "இந்த இணையம் என் உர ادார்மானியை மிகவும் மாற்றிவிட்டது!",
      name: "ராமு, விவசாயி",
      language: "Tamil",
    },
    {
      img: ProfilePic2,
      text: "ఈ వెబ్ అనువర్తనం నా ఎరువు నిర్వహణను విప్లవం చేసింది.",
      name: "వెంకట్, రైతు",
      language: "Telugu",
    },
    {
      img: ProfilePic3,
      text: "ಈ ಅನೇಕ್ನಪಪನ ನಮದ ಎೂೕಟದರ ನಿಳವಣಿಗಗೆ ಸಹಾಯ ಮಾಡಿದೆ.",
      name: "ರಮೇಶ್, ರೈತ",
      language: "Kannada",
    },
    {
      img: ProfilePic4,
      text: "ਇਹ ਵੈੱਬ ਐਪ ਮੇਰੇ ਖੇਤੀਬਾੜੀ ਦੇ ਨਤੀਜੇ ਬਹੁਤ ਵਧੀਆ ਬਣਾਏ ਹਨ।",
      name: "ਜਸਪਾਲ ਸਿੰਘ, ਕਿਸਾਨ",
      language: "Punjabi",
    },
    {
      img: ProfilePic5,
      text: "यह वेब एप्लिकेशन मेरे उर्वरक प्रबंधन को पूरी तरह से बदल दिया है।",
      name: "रवि पटेल, किसान",
      language: "Hindi",
    },
    {
      img: ProfilePic6,
      text: "This web app has revolutionized my fertilizer management!",
      name: "John, Farmer",
      language: "English",
    },
    {
      img: ProfilePic7,
      text: "આ વેબ એપ્લિકેશનએ મારું ખાતર મેનેજમેન્ટ બદલાવી નાખ્યું છે.",
      name: "રાજેશભાઈ, ખેડૂત",
      language: "Gujarati",
    },
  ];

  return (
    <div className="work-section-wrapper">
      <div className="work-section-top">
        <p className="primary-subheading">Testimonial</p>
        <h1 className="primary-heading">What They Are Saying</h1>
        <p className="primary-text">
          Hear from Farmers Who've Transformed Their Harvests
        </p>
      </div>
      <div className="testimonial-marquee">
        <div className="marquee-content">
          {testimonials.concat(testimonials).map((testimonial, index) => (
            <div key={index} className="testimonial-item">
              <img src={testimonial.img} alt="" width="100vu" />
              <p>{testimonial.text}</p>
              <div className="testimonials-stars-container">
                <AiFillStar />
                <AiFillStar />
                <AiFillStar />
                <AiFillStar />
                <AiFillStar />
              </div>
              <h2>— {testimonial.name} ({testimonial.language})</h2>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Testimonial;
