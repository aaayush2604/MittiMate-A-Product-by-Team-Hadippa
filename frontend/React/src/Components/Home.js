import React from 'react';
import Navbar from './Navbar';
import BannerBackground from '../Assets/home-banner-background.png';
import BannerImage from '../Assets/home-banner-image-2.png';
import { FiArrowRight } from 'react-icons/fi';

const Home = () =>{
    return <div className='home-container'>
        <Navbar/>
        <div className='home-banner-container'>
            <div className='home-bannerImage-container'>
                <img src={BannerBackground} alt="" />
            </div>
            <div className='home-text-section'>
                <h1 className='primary-heading'>Nurturing Soil, Boosting Yields.</h1>
                <p className='primary-text'>Empowering farmers with personalized, data-driven fertilizer recommendations to boost crop yields, enhance sustainability, and improve income</p>
                <button className='secondary-button'>Form<FiArrowRight/></button>
            </div>
            <div className='home-image-container'>
                <img src={BannerImage} alt="" width="600 vu" />
            </div>

        </div>
    </div>
};

export default Home;
