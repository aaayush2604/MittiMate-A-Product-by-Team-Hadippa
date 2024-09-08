import React, { useState } from "react";
import { Link } from "react-router-dom"; 
import Logo from "../Assets/logo2.png";
import { BsCart2 } from "react-icons/bs";
import { HiOutlineBars3 } from "react-icons/hi2";
import Box from "@mui/material/Box";
import Drawer from "@mui/material/Drawer";
import List from "@mui/material/List";
import Divider from "@mui/material/Divider";
import ListItem from "@mui/material/ListItem";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import HomeIcon from "@mui/icons-material/Home";
import InfoIcon from "@mui/icons-material/Info";
import CommentRoundedIcon from "@mui/icons-material/CommentRounded";
import PhoneRoundedIcon from "@mui/icons-material/PhoneRounded";

const Navbar = () => {
  const [openMenu, setOpenMenu] = useState(false);
  const menuOptions = [
    {
      text: "Home",
      icon: <HomeIcon />,
      link: "/",
    },
    {
      text: "Get Recommendation",
      icon: <InfoIcon />,
      link: "/plustest",
    },
    {
      text: "Get Appointement",
      icon: <CommentRoundedIcon />,
      link: "/form",
    },
    {
      text: "Contact",
      icon: <PhoneRoundedIcon />,
      link: "/contact",
    },
  ];

  return (
    <nav>
      <div className="nav-logo-container">
        <Link to="/">
          <img src={Logo} alt="" width="100vu" />
        </Link>
      </div>
      <div className="navbar-links-container">
        <Link to="/">Home</Link>
        <Link to="/plustest">Get Recommendation</Link>
        <Link to="/form">Get Appointment</Link>
        <Link to="/contact">Contact</Link>
        <Link to="/signup"><button className="primary-button">Sign Up</button></Link>
      </div>
      <div className="navbar-menu-container">
        <HiOutlineBars3 onClick={() => setOpenMenu(true)} />
      </div>
      <Drawer open={openMenu} onClose={() => setOpenMenu(false)} anchor="right">
        <Box
          sx={{ width: 250 }}
          role="presentation"
          onClick={() => setOpenMenu(false)}
          onKeyDown={() => setOpenMenu(false)}
        >
          <List>
            {menuOptions.map((item) => (
              <ListItem key={item.text} disablePadding>
                <ListItemButton component={Link} to={item.link}>
                  <ListItemIcon>{item.icon}</ListItemIcon>
                  <ListItemText primary={item.text} />
                </ListItemButton>
              </ListItem>
            ))}
          </List>
        </Box>
      </Drawer>
    </nav>
  );
};

export default Navbar;
