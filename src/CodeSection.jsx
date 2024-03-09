import "./CodeSection.css";
import {useRef, useEffect} from 'react';
function CodeSection(props){
  const codeRef=useRef(null);
  let Content;
  useEffect(()=>{
    if(codeRef.current){
       Content=codeRef.current.innerText;
    }

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
</div>
<div className="Code" ref={codeRef}>
{props.code}
</div>
    </div>
  );
}
export default CodeSection;