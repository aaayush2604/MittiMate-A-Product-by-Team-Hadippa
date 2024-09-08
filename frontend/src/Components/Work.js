import React from 'react';
import FillForm from '../Assets/fill-form-image.png';
import Processing from '../Assets/processing-image.png';
import Result from '../Assets/result-image.png';

const Work =()=>{
    const workInfoData =[
        {
            image: FillForm,
            title: "Fill the Form",
            text: "Share Your Farm Details",
        },
        {
            image: Processing,
            title: "Processing",
            text: "AI Analyzes, Solutions Brew",
        },
        {
            image: Result,
            title: "Simplified Report",
            text: "Get Clear Fertilizer Guidance",
        },
    ];
    return (
        <div className="work-section-wrapper">
          <div className="work-section-top">
            <p className="primary-subheading">Smart Fertilizer Predictions with Machine Learning</p>
            <h1 className="primary-heading">How it works?</h1>
            <p className="primary-text">
            Our system leverages advanced machine learning algorithms to analyze soil data, crop types, and weather patterns, delivering precise fertilizer recommendations that optimize growth and sustainability.
            </p>
          </div>
          <div className="work-section-bottom">
            {workInfoData.map((data) => (
              <div className="work-section-info" key={data.title}>
                <div className="info-boxes-img-container">
                  <img src={data.image} alt="" />
                </div>
                <h2>{data.title}</h2>
                <p>{data.text}</p>
              </div>
            ))}
          </div>
        </div>
      );
}

export default Work;