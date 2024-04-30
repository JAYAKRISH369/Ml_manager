import React from "react";
import  ReactDOM from "react-dom/client";
import CreateSection from "./CreateSection.jsx"
import ans from "./answers.jsx";
import Footer from "./Footer.jsx"
ReactDOM.createRoot(document.getElementById('root')).render(
    <div>
    <button><a href="./finddataset.csv" download>find-s data set</a></button>
    <button><a href="./finddataset.csv" download>Candidate Elimination data set</a></button>
    <button><a href="./tennis.csv" download>ID 3 data set</a></button>
 {ans.map(CreateSection)}
 
 <Footer />
 </div>
);