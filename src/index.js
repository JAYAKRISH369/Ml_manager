import React from "react";
import  ReactDOM from "react-dom/client";
import CreateSection from "./CreateSection.jsx"
import ans from "./answers.jsx";
import Footer from "./Footer.jsx"
ReactDOM.createRoot(document.getElementById('root')).render(
    <div>
    <Footer />
 {ans.map(CreateSection)}

 <Footer />
 </div>
);