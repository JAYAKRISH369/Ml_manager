import "./CodeSection.css";
import clipboard_img from "../src/clipboard.png";
import {useRef, useEffect} from 'react';


function CodeSection(props){
  const codeRef=useRef(null);
  let Content;
  useEffect(()=>{
    if(codeRef.current){
       Content=codeRef.current.innerText;
    }
    console.log('called use effect')
  },[]);
  function copyToClipboard() {
    var tempInput = document.createElement("input");
    tempInput.value = Content;
    document.body.appendChild(tempInput);
    tempInput.select();
    document.execCommand("copy");
    document.body.removeChild(tempInput);
  
    alert("Content copied to clipboard!");
  }

  return(
    <div className="Section" onClick={copyToClipboard}>
<div className="Title">
{` ${props.name}`}

{/* <div className="copy-div">
<img src={clipboard_img} alt="clipboard"/>
<button id="copyButton" onClick={copyToClipboard}>copy</button>
</div> */}
</div>
<div className="Code" ref={codeRef}>
{props.code}
</div>
    </div>
  );
}
export default CodeSection;