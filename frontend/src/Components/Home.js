import React from 'react';
import BannerBackground from '../Assets/home-banner-background.png';
import BannerImage from '../Assets/home-banner-image-2.png';
import { FiArrowRight } from 'react-icons/fi';
import About from './About';
import Work from './Work';
import Testimonial from './Testimonial';
import Contact from './Contact';
import { Link } from "react-router-dom"; 
import ChatBot from './ChatBot';

const Home = () =>{
    return <div className='home-container'>
        <div className='home-banner-container'>
            <div className='home-bannerImage-container'>
                <img src={BannerBackground} alt="" />
            </div>
            <div className='home-text-section'>
                <h1 className='primary-heading'>Nurturing Soil, Boosting Yields.</h1>
                <p className='primary-text'>Empowering farmers with personalized, data-driven fertilizer recommendations to boost crop yields, enhance sustainability, and improve income</p>
                <Link className='button123' to="/plustest"><button className='secondary-button'>Get Fertiliser Recommendation<FiArrowRight/></button></Link>
            </div>
            <div className='home-image-container'>
                <img src={BannerImage} alt="" width="600 vu" />
            </div>

        </div>
        <About/>
        <Work/>
        <Testimonial/>
        <Contact/>
        <ChatBot/>

    </div>
};

export default Home;
